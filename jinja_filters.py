import sympy as sp
import jinja2
import copy
from pystencils.astnodes import ResolvedFieldAccess
from pystencils.backends.cbackend import CustomSympyPrinter
from pystencils.types import getBaseType
from pystencils.cpu import generateC


temporaryFieldTemplate = """
// Getting temporary field {tmpFieldName}
static std::set< {type} *, field::SwapableCompare< {type} * > > cache_{originalFieldName};
auto it = cache_{originalFieldName}.find( {originalFieldName} );
{type} * {tmpFieldName};
if( it != cache_{originalFieldName}.end() )
{{
    {tmpFieldName} = *it;
}}
else 
{{
    {tmpFieldName} = {originalFieldName}->cloneUninitialized();
    cache_{originalFieldName}.insert({tmpFieldName});
}}
"""


@jinja2.contextfilter
def generateDeclaration(ctx, kernelInfo):
    """Generates the declaration of the kernel function"""
    isGpu = ctx['target'] == 'gpu'
    ast = kernelInfo.ast
    if isGpu:
        paramsInConstantMem = [p for p in ast.parameters if p.isFieldStrideArgument or p.isFieldShapeArgument]
        ast.globalVariables.update([p.name for p in paramsInConstantMem])

    result = generateC(ast, signatureOnly=True) + ";"
    result = "namespace internal {\n%s\n}" % (result,)
    return result


@jinja2.contextfilter
def generateDefinition(ctx, kernelInfo):
    """Generates the definition (i.e. implementation) of the kernel function"""
    isGpu = ctx['target'] == 'gpu'
    ast = kernelInfo.ast
    if isGpu:
        paramsInConstantMem = [p for p in ast.parameters if p.isFieldStrideArgument or p.isFieldShapeArgument]
        ast = copy.deepcopy(ast)
        ast.globalVariables.update([p.symbol for p in paramsInConstantMem])
        prefix = ["__constant__ %s %s[4];" % (getBaseType(p.dtype).baseName, p.name) for p in paramsInConstantMem]
        prefix = "\n".join(prefix)
    else:
        prefix = ""

    result = generateC(ast)
    result = "namespace internal {\n%s\n%s\n}" % (prefix, result)
    return result


@jinja2.contextfilter
def generateBlockDataToFieldExtraction(ctx, kernelInfo, parametersToIgnore=[]):
    """Generates code that either extracts the fields from a block or uses temporary fields that are swapped later"""
    ast = kernelInfo.ast
    fields = {f.name: f for f in ast.fieldsAccessed}
    fieldAccesses = ast.atoms(ResolvedFieldAccess)

    def makeFieldType(dtype, fSize):
        if ctx['target'] == 'cpu':
            return "GhostLayerField<%s, %d>" % (dtype, fSize)
        else:
            return "cuda::GPUField<%s>" % (dtype,)

    def getMaxIndexCoordinateForField(fieldName):
        field = fields[fieldName]
        if field.indexDimensions == 0:
            return 1
        else:
            maxIdxValue = 0
            for acc in fieldAccesses:
                if acc.field == field and acc.idxCoordinateValues[0] > maxIdxValue:
                    maxIdxValue = acc.idxCoordinateValues[0]
            return maxIdxValue

    result = []
    toIgnore = parametersToIgnore.copy() + kernelInfo.temporaryFields
    for param in ast.parameters:
        if param.isFieldPtrArgument and param.fieldName not in toIgnore:
            dType = getBaseType(param.dtype)
            fSize = getMaxIndexCoordinateForField(param.fieldName)
            result.append("auto %s = block->getData< %s >(%sID);" %
                          (param.fieldName, makeFieldType(dType, fSize), param.fieldName))

    for tmpFieldName in kernelInfo.temporaryFields:
        if tmpFieldName in parametersToIgnore:
            continue
        assert tmpFieldName.endswith('_tmp')
        originalFieldName = tmpFieldName[:-len('_tmp')]
        elementType = getBaseType(fields[originalFieldName].dtype)
        dtype = makeFieldType(elementType, getMaxIndexCoordinateForField(originalFieldName))
        result.append(temporaryFieldTemplate.format(originalFieldName=originalFieldName,
                                                    tmpFieldName=tmpFieldName,
                                                    type=dtype))

    return "\n".join(result)


@jinja2.contextfilter
def generateCall(ctx, kernelInfo):
    """Generates the function call to a pystencils kernel"""
    ast = kernelInfo.ast

    isCpu = ctx['target'] == 'cpu'

    kernelCallLines = []
    fields = {f.name: f for f in ast.fieldsAccessed}

    spatialShapeSymbols = []

    for param in ast.parameters:
        typeStr = getBaseType(param.dtype).baseName
        if param.isFieldPtrArgument:
            kernelCallLines.append("%s %s = (%s *) %s->data();" % (param.dtype, param.name, typeStr, param.fieldName))
        elif param.isFieldStrideArgument:
            strideNames = ['xStride()', 'yStride()', 'zStride()', 'fStride()']
            strideNames = ["%s(%s->%s)" % (typeStr, param.fieldName, e) for e in strideNames]
            field = fields[param.fieldName]
            strides = strideNames[:field.spatialDimensions]
            assert field.indexDimensions in (0, 1)
            if field.indexDimensions == 1:
                strides.append(strideNames[-1])
            if isCpu:
                kernelCallLines.append("const %s %s [] = {%s};" % (typeStr, param.name, ", ".join(strides)))
            else:
                kernelCallLines.append("const %s %s_cpu [] = {%s};" % (typeStr, param.name, ", ".join(strides)))
                kernelCallLines.append("cudaMemcpyToSymbol(internal::%s, %s_cpu, %d * sizeof(%s));"
                                       % (param.name, param.name, len(strides), typeStr))

        elif param.isFieldShapeArgument:
            shapeNames = ['xSizeWithGhostLayer()', 'ySizeWithGhostLayer()', 'zSizeWithGhostLayer()', 'fSize()']
            typeStr = getBaseType(param.dtype).baseName
            shapeNames = ["%s(%s->%s)" % (typeStr, param.fieldName, e) for e in shapeNames]
            field = fields[param.fieldName]
            shapes = shapeNames[:field.spatialDimensions]

            spatialShapeSymbols = [sp.Symbol("%s_cpu[%d]" % (param.name, i)) for i in range(field.spatialDimensions)]

            assert field.indexDimensions in (0, 1)
            if field.indexDimensions == 1:
                shapes.append(shapeNames[-1])
            if isCpu:
                kernelCallLines.append("const %s %s [] = {%s};" % (typeStr, param.name, ", ".join(shapes)))
            else:
                kernelCallLines.append("const %s %s_cpu [] = {%s};" % (typeStr, param.name, ", ".join(shapes)))
                kernelCallLines.append("cudaMemcpyToSymbol(internal::%s, %s_cpu, %d * sizeof(%s));"
                                       % (param.name, param.name, len(shapes), typeStr))

    if not isCpu:
        indexingDict = ast.indexing.getCallParameters(spatialShapeSymbols)
        callParameters = ", ".join([p.name for p in ast.parameters if p.isFieldPtrArgument or not p.isFieldArgument])
        spPrinterC = CustomSympyPrinter()

        kernelCallLines += [
            "dim3 _block(int(%s), int(%s), int(%s));" % tuple(spPrinterC.doprint(e) for e in indexingDict['block']),
            "dim3 _grid(int(%s), int(%s), int(%s));" % tuple(spPrinterC.doprint(e) for e in indexingDict['grid']),
            "internal::%s<<<_grid, _block>>>(%s);" % (ast.functionName, callParameters),
        ]
    else:
        kernelCallLines.append("internal::%s(%s);" % (ast.functionName, ", ".join([p.name for p in ast.parameters])))
    return "\n".join(kernelCallLines)


def generateSwaps(kernelInfo):
    """Generates code to swap main fields with temporary fields"""
    swaps = ""
    for src, dst in kernelInfo.fieldSwaps:
        swaps += "%s->swapDataPointers(%s);\n" % (src, dst)
    return swaps


def generateConstructorInitializerList(kernelInfo, parametersToIgnore=[]):
    ast = kernelInfo.ast
    parametersToIgnore += kernelInfo.temporaryFields

    parameterInitializerList = []
    for param in ast.parameters:
        if param.isFieldPtrArgument and param.fieldName not in parametersToIgnore:
            parameterInitializerList.append("%sID(%sID_)" % (param.fieldName, param.fieldName))
        elif not param.isFieldArgument and param.name not in parametersToIgnore:
            parameterInitializerList.append("%s(%s_)" % (param.name, param.name))
    return ", ".join(parameterInitializerList)


def generateConstructorParameters(kernelInfo, parametersToIgnore=[]):
    ast = kernelInfo.ast
    parametersToIgnore += kernelInfo.temporaryFields

    parameterList = []
    for param in ast.parameters:
        if param.isFieldPtrArgument and param.fieldName not in parametersToIgnore:
            parameterList.append("BlockDataID %sID_" % (param.fieldName, ))
        elif not param.isFieldArgument and param.name not in parametersToIgnore:
            parameterList.append("%s %s_" % (param.dtype, param.name,))
    return ", ".join(parameterList)


def generateMembers(kernelInfo, parametersToIgnore=[]):
    ast = kernelInfo.ast
    parametersToIgnore += kernelInfo.temporaryFields

    result = []
    for param in ast.parameters:
        if param.isFieldPtrArgument and param.fieldName not in parametersToIgnore:
            result.append("BlockDataID %sID;" % (param.fieldName, ))
        elif not param.isFieldArgument and param.name not in parametersToIgnore:
            result.append("%s %s;" % (param.dtype, param.name,))
    return "\n".join(result)


def addPystencilsFiltersToJinjaEnv(jinjaEnv):
    jinjaEnv.filters['generateDefinition'] = generateDefinition
    jinjaEnv.filters['generateDeclaration'] = generateDeclaration
    jinjaEnv.filters['generateMembers'] = generateMembers
    jinjaEnv.filters['generateConstructorParameters'] = generateConstructorParameters
    jinjaEnv.filters['generateConstructorInitializerList'] = generateConstructorInitializerList
    jinjaEnv.filters['generateCall'] = generateCall
    jinjaEnv.filters['generateBlockDataToFieldExtraction'] = generateBlockDataToFieldExtraction
    jinjaEnv.filters['generateSwaps'] = generateSwaps

