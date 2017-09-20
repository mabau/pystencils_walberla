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


def fieldExtractionCode(fieldAccesses, fieldName, isTemporary, declarationOnly=False, noDeclaration=False, isGpu=False):
    """
    Returns code string for getting a field pointer.
    This can happen in two ways: either the field is extracted from a waLBerla block, or a temporary field to swap is
    created.

    :param fieldAccesses: set of Field.Access objects of a kernel
    :param fieldName: the field name for which the code should be created
    :param isTemporary: extract field from block (False) or create a temporary copy of an existing field (True)
    :param declarationOnly: only create declaration instead of the full code
    :param noDeclaration: create the extraction code, and assume that declarations are elsewhere
    :param isGpu: if the field is a GhostLayerField or a GpuField
    """
    fields = {fa.field.name: fa.field for fa in fieldAccesses}
    field = fields[fieldName]

    def makeFieldType(dtype, fSize):
        if isGpu:
            return "cuda::GPUField<%s>" % (dtype,)
        else:
            return "GhostLayerField<%s, %d>" % (dtype, fSize)

    # Determine size of f coordinate which is a template parameter
    if field.indexDimensions == 0:
        fSize = 1
    else:
        maxIdxValue = 0
        for acc in fieldAccesses:
            if acc.field == field and acc.idxCoordinateValues[0] > maxIdxValue:
                maxIdxValue = acc.idxCoordinateValues[0]
        fSize = maxIdxValue + 1

    dtype = getBaseType(field.dtype)
    fieldType = "cuda::GPUField<%s>" % (dtype,) if isGpu else "GhostLayerField<%s, %d>" % (dtype, fSize)

    if not isTemporary:
        dType = getBaseType(field.dtype)
        fieldType = makeFieldType(dType, fSize)
        if declarationOnly:
            return "%s * %s;" % (fieldType, fieldName)
        else:
            prefix = "" if noDeclaration else "auto "
            return "%s%s = block->getData< %s >(%sID);" % (prefix, fieldName, fieldType, fieldName)
    else:
        assert fieldName.endswith('_tmp')
        originalFieldName = fieldName[:-len('_tmp')]
        if declarationOnly:
            return "%s * %s;" % (fieldType, fieldName)
        else:
            declaration = "{type} * {tmpFieldName};".format(type=fieldType, tmpFieldName=fieldName)
            tmpFieldStr = temporaryFieldTemplate.format(originalFieldName=originalFieldName,
                                                        tmpFieldName=fieldName,
                                                        type=fieldType)
            return tmpFieldStr if noDeclaration else declaration + tmpFieldStr


@jinja2.contextfilter
def generateBlockDataToFieldExtraction(ctx, kernelInfo, parametersToIgnore=[], parameters=None,
                                       declarationsOnly=False, noDeclarations=False):
    ast = kernelInfo.ast
    fieldAccesses = ast.atoms(ResolvedFieldAccess)

    if parameters is not None:
        assert parametersToIgnore == []
    else:
        parameters = {p.fieldName for p in ast.parameters if p.isFieldPtrArgument}
        parameters.difference_update(parametersToIgnore)

    normal = {f for f in parameters if f not in kernelInfo.temporaryFields}
    temporary = {f for f in parameters if f in kernelInfo.temporaryFields}

    args = {
        'fieldAccesses': fieldAccesses,
        'declarationOnly': declarationsOnly,
        'noDeclaration': noDeclarations,
        'isGpu': ctx['target'] == 'gpu',
    }
    result = "\n".join(fieldExtractionCode(fieldName=fn, isTemporary=False, **args) for fn in normal) + "\n"
    result += "\n".join(fieldExtractionCode(fieldName=fn, isTemporary=True, **args) for fn in temporary)
    return result


@jinja2.contextfilter
def generateBlockDataToFieldExtractionOld(ctx, kernelInfo, parametersToIgnore=[],
                                       declarationsOnly=False, noDeclarations=False):
    """Generates code that either extracts the fields from a block or uses temporary fields that are swapped later"""
    ast = kernelInfo.ast
    fieldAccesses = ast.atoms(ResolvedFieldAccess)
    fields = {f.name: f for f in fieldAccesses}

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
            return maxIdxValue + 1

    result = []
    toIgnore = parametersToIgnore.copy() + kernelInfo.temporaryFields
    for param in ast.parameters:
        if param.isFieldPtrArgument and param.fieldName not in toIgnore:
            dType = getBaseType(param.dtype)
            fSize = getMaxIndexCoordinateForField(param.fieldName)
            fieldType = makeFieldType(dType, fSize)
            if not declarationsOnly:
                prefix = "" if noDeclarations else "auto "
                result.append("%s%s = block->getData< %s >(%sID);" %
                              (prefix, param.fieldName, fieldType, param.fieldName))
            else:
                result.append("%s * %s;" % (fieldType, param.fieldName))

    for tmpFieldName in kernelInfo.temporaryFields:
        if tmpFieldName in parametersToIgnore:
            continue
        assert tmpFieldName.endswith('_tmp')
        originalFieldName = tmpFieldName[:-len('_tmp')]
        elementType = getBaseType(fields[originalFieldName].dtype)
        dtype = makeFieldType(elementType, getMaxIndexCoordinateForField(originalFieldName))
        if not declarationsOnly:
            declaration = "{type} * {tmpFieldName};".format(type=dtype, tmpFieldName=tmpFieldName)
            tmpFieldStr = temporaryFieldTemplate.format(originalFieldName=originalFieldName,
                                                        tmpFieldName=tmpFieldName,
                                                        type=dtype)
            if noDeclarations:
                result.append(tmpFieldStr)
            else:
                result.append(declaration)
                result.append(tmpFieldStr)
        else:
            result.append("%s * %s;" % (dtype, tmpFieldName))

    return "\n".join(result)


def generateRefsForKernelParameters(kernelInfo, prefix, parametersToIgnore):
    symbols = {p.fieldName for p in kernelInfo.ast.parameters if p.isFieldPtrArgument}
    symbols.update(p.name for p in kernelInfo.ast.parameters if not p.isFieldArgument)
    symbols.difference_update(parametersToIgnore)
    return "\n".join("auto & %s = %s%s;" % (s, prefix, s) for s in symbols)


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
            # Correction for non 3D fields - ghost layers in non-handled directions have to be skipped
            # e.g. for 2D kernels the z ghost layers have to be skipped
            field = fields[param.fieldName]
            strideNames = ['xStride()', 'yStride()', 'zStride()', 'fStride()']
            ghostLayerOffsets = []
            for i in range(field.spatialDimensions, 3):
                ghostLayerOffsets.append("%s->%s" % (param.fieldName, strideNames[i]))
            ghostLayerOffsetStr = " + " + " + ".join(ghostLayerOffsets) if ghostLayerOffsets else ""
            kernelCallLines.append("%s %s = (%s *) (%s->data()) %s;" % (param.dtype, param.name, typeStr,
                                                                       param.fieldName, ghostLayerOffsetStr))
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
    jinjaEnv.filters['generateRefsForKernelParameters'] = generateRefsForKernelParameters
