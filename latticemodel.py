import sympy as sp
from lbmpy.creationfunctions import createLatticeBoltzmannMethod, updateWithDefaultParameters
from jinja2 import Environment, PackageLoader


def generateLatticeModel(latticeModelName, optimizationParams={}, **kwargs):
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

    stencilName = params['stencil']
    relaxationRates = params['relaxationRates']

    for rr in relaxationRates:
        if not isinstance(rr, sp.Symbol):
            raise ValueError("Only symbolic relaxation rates supported (have to be adapted to refinement level)")
    for f in params['force']:
        if not isinstance(rr, sp.Symbol) and rr != 0:
            raise ValueError("Only symbolic force values are supported (have to be adapted to refinement level)")

    method = createLatticeBoltzmannMethod(**params)

    context = {
        'modelName': latticeModelName,
        'stencilName': stencilName,
        'Q': len(method.stencil),
        'compressible': 'true' if params['compressible'] else 'false',
        'weights': ",".join(str(w.evalf()) for w in method.weights),
        'inverseWeights': ",".join(str((1/w).evalf()) for w in method.weights),
        'relaxationRates': [rr.name for rr in relaxationRates],
        'forceParameters': [f.name for f in params['force'] if f != 0],
    }

    env = Environment(loader=PackageLoader('pystencils_walberla'))

    renderedTmpl = env.get_template('LatticeModel.tmpl.h').render(**context)
    print(renderedTmpl)


if __name__ == '__main__':
    generateLatticeModel(latticeModelName='TestModel', method='srt', stencil='D3Q19')
