import sympy as sp
from sympy.tensor import IndexedBase
from jinja2 import Environment, PackageLoader, Template

from pystencils.astnodes import SympyAssignment
from pystencils.sympyextensions import getSymmetricPart
from pystencils.field import offsetToDirectionString
from pystencils.backends.cbackend import CustomSympyPrinter, CBackend
from pystencils.types import TypedSymbol
from pystencils_walberla.sweep import KernelInfo
from pystencils_walberla.jinja_filters import addPystencilsFiltersToJinjaEnv

from lbmpy.creationfunctions import createLatticeBoltzmannMethod, updateWithDefaultParameters,\
    createLatticeBoltzmannAst, createLatticeBoltzmannUpdateRule
from lbmpy.updatekernels import createStreamPullOnlyKernel

cppPrinter = CustomSympyPrinter()


def stencilSwitchStatement(stencil, values):
    templ = Template("""
    using namespace stencil;
    switch( direction ) {
        {% for directionName, value in dirToValueDict.items() -%}
            case {{directionName}}: return {{value}};
        {% endfor -%}
    }
    """)

    dirToValueDict = {offsetToDirectionString(d): cppPrinter.doprint(v) for d, v in zip(stencil, values)}
    return templ.render(dirToValueDict=dirToValueDict)


def equationsToCode(equations):
    def typeEq(eq):
        return eq.subs({s: TypedSymbol(s.name, "real_t") for s in eq.atoms(sp.Symbol)})

    cBackend = CBackend()
    result = []
    for eq in equations:
        assignment = SympyAssignment(typeEq(eq.lhs), typeEq(eq.rhs))
        result.append(cBackend(assignment))
    return "\n".join(result)


def generateLatticeModel(latticeModelName, optimizationParams={}, **kwargs):
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

    stencilName = params['stencil']
    relaxationRates = params['relaxationRates']

    if params['forceModel'] != 'none' and params['force'] == (0, 0, 0):
        params['force'] = sp.symbols("force:3")

    for rr in relaxationRates:
        if not isinstance(rr, sp.Symbol):
            raise ValueError("Only symbolic relaxation rates supported (have to be adapted to refinement level)")
    for f in params['force']:
        if not isinstance(rr, sp.Symbol) and rr != 0:
            raise ValueError("Only symbolic force values are supported (have to be adapted to refinement level)")

    params['fieldName'] = 'pdfs'
    params['secondFieldName'] = 'pdfs_tmp'

    method = createLatticeBoltzmannMethod(**params)
    streamCollideUpdate = createLatticeBoltzmannUpdateRule(lbMethod=method, optimizationParams=optParams, **params)
    streamCollideAst = createLatticeBoltzmannAst(updateRule=streamCollideUpdate, optimizationParams=optParams, **params)

    params['kernelType'] = 'collideOnly'
    collideOnlyUpdate = createLatticeBoltzmannUpdateRule(lbMethod=method, optimizationParams=optParams, **params)
    collideAst = createLatticeBoltzmannAst(updateRule=collideOnlyUpdate, optimizationParams=optParams, **params)

    #streamOnlyAst = createStreamPullOnlyKernel(method.stencil)

    velSymbols = method.conservedQuantityComputation.firstOrderMomentSymbols
    velArrSymbols = [IndexedBase(sp.Symbol('u'), shape=(1,))[i] for i in range(len(velSymbols))]

    equilibrium = method.getEquilibriumTerms().subs({a: b for a, b in zip(velSymbols, velArrSymbols)})
    symmetricEquilibrium = getSymmetricPart(equilibrium, velArrSymbols)
    asymmetricEquilibrium = sp.expand(equilibrium - symmetricEquilibrium)

    forceModel = method.forceModel
    macroscopicVelocityShift = None
    if forceModel:
        if hasattr(forceModel, 'macroscopicVelocityShift'):
            macroscopicVelocityShift = [cppPrinter.doprint(e)
                                        for e in forceModel.macroscopicVelocityShift(sp.Symbol("rho"))]

    cqc = method.conservedQuantityComputation
    densityOut = cqc.outputEquationsFromPdfs(sp.symbols('f_:19'), {'density': sp.Symbol("rho")})
    context = {
        'className': latticeModelName,
        'stencilName': stencilName,
        'Q': len(method.stencil),
        'compressible': 'true' if params['compressible'] else 'false',
        'weights': ",".join(str(w.evalf()) for w in method.weights),
        'inverseWeights': ",".join(str((1/w).evalf()) for w in method.weights),
        'relaxationRates': [rr.name for rr in relaxationRates],
        #'forceParameters': [f_i.name for f_i in params['force'] if f_i != 0],
        'equilibriumAccuracyOrder': params['equilibriumAccuracyOrder'],

        'equilibriumFromDirection': stencilSwitchStatement(method.stencil, equilibrium),
        'symmetricEquilibriumFromDirection': stencilSwitchStatement(method.stencil, symmetricEquilibrium),
        'asymmetricEquilibriumFromDirection': stencilSwitchStatement(method.stencil, asymmetricEquilibrium),
        'equilibrium': [cppPrinter.doprint(e) for e in equilibrium],

        'macroscopicVelocityShift': macroscopicVelocityShift,
        'densityOut': equationsToCode(densityOut.allEquations),

        'streamCollideKernel': KernelInfo(streamCollideAst, ['pdfs_tmp'], [('pdfs', 'pdfs_tmp')]),
        'target': 'cpu',
    }

    env = Environment(loader=PackageLoader('pystencils_walberla'))
    addPystencilsFiltersToJinjaEnv(env)

    return env.get_template('LatticeModel.tmpl.h').render(**context)


if __name__ == '__main__':
    from pystencils import Field
    forceField = Field.createGeneric('force', spatialDimensions=3, indexDimensions=1, layout='c')
    force = [forceField(0), forceField(1), forceField(2)]
    res = generateLatticeModel(latticeModelName='TestModel', method='srt', stencil='D3Q19',
                               forceModel='guo', force=force)
    with open("result.h", 'w') as f:
        f.write(res)
