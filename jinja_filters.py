import sympy as sp
import jinja2
import copy
from pystencils.astnodes import ResolvedFieldAccess
from pystencils.data_types import get_base_type
from pystencils.backends.cbackend import generate_c, CustomSympyPrinter
from pystencils.field import FieldType


temporary_fieldTemplate = """
// Getting temporary field {tmp_field_name}
static std::set< {type} *, field::SwapableCompare< {type} * > > cache_{original_field_name};
auto it = cache_{original_field_name}.find( {original_field_name} );
if( it != cache_{original_field_name}.end() )
{{
    {tmp_field_name} = *it;
}}
else 
{{
    {tmp_field_name} = {original_field_name}->cloneUninitialized();
    cache_{original_field_name}.insert({tmp_field_name});
}}
"""


@jinja2.contextfilter
def generate_declaration(ctx, kernel_info):
    """Generates the declaration of the kernel function"""
    is_gpu = ctx['target'] == 'gpu'
    ast = kernel_info.ast
    if is_gpu:
        params_in_constant_mem = [p for p in ast.parameters if p.is_field_stride_argument or p.is_field_shape_argument]
        ast.global_variables.update([p.name for p in params_in_constant_mem])

    result = generate_c(ast, signature_only=True) + ";"
    result = "namespace internal {\n%s\n}" % (result,)
    return result


@jinja2.contextfilter
def generate_definition(ctx, kernel_info):
    """Generates the definition (i.e. implementation) of the kernel function"""
    is_gpu = ctx['target'] == 'gpu'
    ast = kernel_info.ast
    if is_gpu:
        params_in_constant_mem = [p for p in ast.parameters if p.is_field_stride_argument or p.is_field_shape_argument]
        ast = copy.deepcopy(ast)
        ast.global_variables.update([p.symbol for p in params_in_constant_mem])
        prefix = ["__constant__ %s %s[4];" % (get_base_type(p.dtype).base_name, p.name) for p in params_in_constant_mem]
        prefix = "\n".join(prefix)
    else:
        prefix = ""

    result = generate_c(ast)
    result = "namespace internal {\n%s\nstatic %s\n}" % (prefix, result)
    return result


def field_extraction_code(field_accesses, field_name, is_temporary, declaration_only=False,
                          no_declaration=False, is_gpu=False):
    """Returns code string for getting a field pointer.

    This can happen in two ways: either the field is extracted from a walberla block, or a temporary field to swap is
    created.

    Args:
        field_accesses: set of Field.Access objects of a kernel
        field_name: the field name for which the code should be created
        is_temporary: new_filtered field from block (False) or create a temporary copy of an existing field (True)
        declaration_only: only create declaration instead of the full code
        no_declaration: create the extraction code, and assume that declarations are elsewhere
        is_gpu: if the field is a GhostLayerField or a GpuField
    """
    fields = {fa.field.name: fa.field for fa in field_accesses}
    field = fields[field_name]

    def make_field_type(dtype, f_size):
        if is_gpu:
            return "cuda::GPUField<%s>" % (dtype,)
        else:
            return "GhostLayerField<%s, %d>" % (dtype, f_size)

    # Determine size of f coordinate which is a template parameter
    assert field.index_dimensions <= 1
    if field.has_fixed_index_shape and field.index_dimensions > 0:
        f_size = field.index_shape[0]
    elif field.index_dimensions == 0:
        f_size = 1
    else:
        max_idx_value = 0
        for acc in field_accesses:
            if acc.field == field and acc.idx_coordinate_values[0] > max_idx_value:
                max_idx_value = acc.idx_coordinate_values[0]
        f_size = max_idx_value + 1

    dtype = get_base_type(field.dtype)
    field_type = "cuda::GPUField<%s>" % (dtype,) if is_gpu else "GhostLayerField<%s, %d>" % (dtype, f_size)

    if not is_temporary:
        dtype = get_base_type(field.dtype)
        field_type = make_field_type(dtype, f_size)
        if declaration_only:
            return "%s * %s;" % (field_type, field_name)
        else:
            prefix = "" if no_declaration else "auto "
            return "%s%s = block->getData< %s >(%sID);" % (prefix, field_name, field_type, field_name)
    else:
        assert field_name.endswith('_tmp')
        original_field_name = field_name[:-len('_tmp')]
        if declaration_only:
            return "%s * %s;" % (field_type, field_name)
        else:
            declaration = "{type} * {tmp_field_name};".format(type=field_type, tmp_field_name=field_name)
            tmp_field_str = temporary_fieldTemplate.format(original_field_name=original_field_name,
                                                           tmp_field_name=field_name, type=field_type)
            return tmp_field_str if no_declaration else declaration + tmp_field_str


@jinja2.contextfilter
def generate_block_data_to_field_extraction(ctx, kernel_info, parameters_to_ignore=[], parameters=None,
                                            declarations_only=False, no_declarations=False):
    ast = kernel_info.ast
    field_accesses = ast.atoms(ResolvedFieldAccess)

    if parameters is not None:
        assert parameters_to_ignore == []
    else:
        parameters = {p.field_name for p in ast.parameters if p.is_field_ptr_argument}
        parameters.difference_update(parameters_to_ignore)

    normal = {f for f in parameters if f not in kernel_info.temporary_fields}
    temporary = {f for f in parameters if f in kernel_info.temporary_fields}

    args = {
        'field_accesses': field_accesses,
        'declaration_only': declarations_only,
        'no_declaration': no_declarations,
        'is_gpu': ctx['target'] == 'gpu',
    }
    result = "\n".join(field_extraction_code(field_name=fn, is_temporary=False, **args) for fn in normal) + "\n"
    result += "\n".join(field_extraction_code(field_name=fn, is_temporary=True, **args) for fn in temporary)
    return result


def generate_refs_for_kernel_parameters(kernel_info, prefix, parameters_to_ignore):
    symbols = {p.field_name for p in kernel_info.ast.parameters if p.is_field_ptr_argument}
    symbols.update(p.name for p in kernel_info.ast.parameters if not p.is_field_argument)
    symbols.difference_update(parameters_to_ignore)
    return "\n".join("auto & %s = %s%s;" % (s, prefix, s) for s in symbols)


@jinja2.contextfilter
def generate_call(ctx, kernel_info, ghost_layers_to_include=0):
    """Generates the function call to a pystencils kernel"""
    ast = kernel_info.ast

    ghost_layers_to_include = sp.sympify(ghost_layers_to_include)
    if ast.ghost_layers is None:
        required_ghost_layers = 0
    else:
        # ghost layer info is ((x_gl_front, x_gl_end), (y_gl_front, y_gl_end).. )
        required_ghost_layers = max(max(ast.ghost_layers))

    is_cpu = ctx['target'] == 'cpu'

    kernel_call_lines = []
    fields = {f.name: f for f in ast.fields_accessed}

    spatial_shape_symbols = []

    for param in ast.parameters:
        if param.is_field_argument and FieldType.is_indexed(fields[param.field_name]):
            continue

        if param.is_field_ptr_argument:
            field = fields[param.field_name]
            coordinates = [-ghost_layers_to_include - required_ghost_layers] * field.spatial_dimensions
            while len(coordinates) < 4:
                coordinates.append(0)

            actual_gls = sp.Symbol(param.field_name + "->nrOfGhostLayers()") - ghost_layers_to_include
            if required_ghost_layers > 0:
                kernel_call_lines.append("WALBERLA_CHECK_GREATER_EQUAL(%s, %s);" % (actual_gls, required_ghost_layers))
            kernel_call_lines.append("%s %s = %s->dataAt(%s, %s, %s, %s);" %
                                     ((param.dtype, param.name, param.field_name) + tuple(coordinates)))

        elif param.is_field_stride_argument:
            type_str = get_base_type(param.dtype).base_name
            stride_names = ['xStride()', 'yStride()', 'zStride()', 'fStride()']
            stride_names = ["%s(%s->%s)" % (type_str, param.field_name, e) for e in stride_names]
            field = fields[param.field_name]
            strides = stride_names[:field.spatial_dimensions]
            assert field.index_dimensions in (0, 1)
            if field.index_dimensions == 1:
                strides.append(stride_names[-1])
            if is_cpu:
                kernel_call_lines.append("const %s %s [] = {%s};" % (type_str, param.name, ", ".join(strides)))
            else:
                kernel_call_lines.append("const %s %s_cpu [] = {%s};" % (type_str, param.name, ", ".join(strides)))
                kernel_call_lines.append("cudaMemcpyToSymbol(internal::%s, %s_cpu, %d * sizeof(%s));"
                                         % (param.name, param.name, len(strides), type_str))

        elif param.is_field_shape_argument:
            offset = 2 * ghost_layers_to_include + 2 * required_ghost_layers
            shape_names = ['xSize()', 'ySize()', 'zSize()', 'fSize()']
            type_str = get_base_type(param.dtype).base_name
            shape_names = ["%s(%s->%s + %s)" % (type_str, param.field_name, e, offset) for e in shape_names]
            field = fields[param.field_name]
            shapes = shape_names[:field.spatial_dimensions]

            spatial_shape_symbols = [sp.Symbol("%s_cpu[%d]" % (param.name, i)) for i in range(field.spatial_dimensions)]

            assert field.index_dimensions in (0, 1)
            if field.index_dimensions == 1:
                shapes.append(shape_names[-1])
            if is_cpu:
                kernel_call_lines.append("const %s %s [] = {%s};" % (type_str, param.name, ", ".join(shapes)))
            else:
                kernel_call_lines.append("const %s %s_cpu [] = {%s};" % (type_str, param.name, ", ".join(shapes)))
                kernel_call_lines.append("cudaMemcpyToSymbol(internal::%s, %s_cpu, %d * sizeof(%s));"
                                         % (param.name, param.name, len(shapes), type_str))

    if not is_cpu:
        indexing_dict = ast.indexing.call_parameters(spatial_shape_symbols)
        call_parameters = ", ".join([p.name for p in ast.parameters
                                     if p.is_field_ptr_argument or not p.is_field_argument])
        sp_printer_c = CustomSympyPrinter()

        kernel_call_lines += [
            "dim3 _block(int(%s), int(%s), int(%s));" % tuple(sp_printer_c.doprint(e) for e in indexing_dict['block']),
            "dim3 _grid(int(%s), int(%s), int(%s));" % tuple(sp_printer_c.doprint(e) for e in indexing_dict['grid']),
            "internal::%s<<<_grid, _block>>>(%s);" % (ast.function_name, call_parameters),
        ]
    else:
        kernel_call_lines.append("internal::%s(%s);" % (ast.function_name, ", ".join([p.name for p in ast.parameters])))
    return "\n".join(kernel_call_lines)


def generate_swaps(kernel_info):
    """Generates code to swap main fields with temporary fields"""
    swaps = ""
    for src, dst in kernel_info.field_swaps:
        swaps += "%s->swapDataPointers(%s);\n" % (src, dst)
    return swaps


def generate_constructor_initializer_list(kernel_info, parameters_to_ignore=[]):
    ast = kernel_info.ast
    parameters_to_ignore += kernel_info.temporary_fields

    parameter_initializer_list = []
    for param in ast.parameters:
        if param.is_field_ptr_argument and param.field_name not in parameters_to_ignore:
            parameter_initializer_list.append("%sID(%sID_)" % (param.field_name, param.field_name))
        elif not param.is_field_argument and param.name not in parameters_to_ignore:
            parameter_initializer_list.append("%s(%s_)" % (param.name, param.name))
    return ", ".join(parameter_initializer_list)


def generate_constructor_parameters(kernel_info, parameters_to_ignore=[]):
    ast = kernel_info.ast
    parameters_to_ignore += kernel_info.temporary_fields

    parameter_list = []
    for param in ast.parameters:
        if param.is_field_ptr_argument and param.field_name not in parameters_to_ignore:
            parameter_list.append("BlockDataID %sID_" % (param.field_name, ))
        elif not param.is_field_argument and param.name not in parameters_to_ignore:
            parameter_list.append("%s %s_" % (param.dtype, param.name,))
    return ", ".join(parameter_list)


def generate_members(kernel_info, parameters_to_ignore=[]):
    ast = kernel_info.ast
    parameters_to_ignore += kernel_info.temporary_fields

    result = []
    for param in ast.parameters:
        if param.is_field_ptr_argument and param.field_name not in parameters_to_ignore:
            result.append("BlockDataID %sID;" % (param.field_name, ))
        elif not param.is_field_argument and param.name not in parameters_to_ignore:
            result.append("%s %s;" % (param.dtype, param.name,))
    return "\n".join(result)


def add_pystencils_filters_to_jinja_env(jinja_env):
    jinja_env.filters['generate_definition'] = generate_definition
    jinja_env.filters['generate_declaration'] = generate_declaration
    jinja_env.filters['generate_members'] = generate_members
    jinja_env.filters['generate_constructor_parameters'] = generate_constructor_parameters
    jinja_env.filters['generate_constructor_initializer_list'] = generate_constructor_initializer_list
    jinja_env.filters['generate_call'] = generate_call
    jinja_env.filters['generate_block_data_to_field_extraction'] = generate_block_data_to_field_extraction
    jinja_env.filters['generate_swaps'] = generate_swaps
    jinja_env.filters['generate_refs_for_kernel_parameters'] = generate_refs_for_kernel_parameters
