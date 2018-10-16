import sympy as sp
import jinja2
import copy
from pystencils.astnodes import ResolvedFieldAccess
from pystencils.data_types import get_base_type
from pystencils.backends.cbackend import generate_c, CustomSympyPrinter
from pystencils.field import FieldType
from pystencils.sympyextensions import prod

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


def make_field_type(dtype, f_size, is_gpu):
    if is_gpu:
        return "cuda::GPUField<%s>" % (dtype,)
    else:
        return "GhostLayerField<%s, %d>" % (dtype, f_size)


def get_field_fsize(field, field_accesses=()):
    if field.has_fixed_index_shape and field.index_dimensions > 0:
        return prod(field.index_shape)
    elif field.index_dimensions == 0:
        return 1
    else:
        assert field.index_dimensions == 1
        max_idx_value = 0
        for acc in field_accesses:
            if acc.field == field and acc.idx_coordinate_values[0] > max_idx_value:
                max_idx_value = acc.idx_coordinate_values[0]
        return max_idx_value + 1

@jinja2.contextfilter
def generate_declaration(ctx, kernel_info):
    """Generates the declaration of the kernel function"""
    is_gpu = ctx['target'] == 'gpu'
    ast = kernel_info.ast
    if is_gpu:
        params_in_constant_mem = [p for p in ast.parameters if p.is_field_stride_argument or p.is_field_shape_argument]
        ast.global_variables.update([p.name for p in params_in_constant_mem])

    result = generate_c(ast, signature_only=True) + ";"
    result = "namespace internal_%s {\n%s\n}" % (ast.function_name, result,)
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
    result = "namespace internal_%s {\n%s\nstatic %s\n}" % (ast.function_name, prefix, result)
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

    # Determine size of f coordinate which is a template parameter
    f_size = get_field_fsize(field, field_accesses)
    dtype = get_base_type(field.dtype)
    field_type = "cuda::GPUField<%s>" % (dtype,) if is_gpu else "GhostLayerField<%s, %d>" % (dtype, f_size)

    if not is_temporary:
        dtype = get_base_type(field.dtype)
        field_type = make_field_type(dtype, f_size, is_gpu)
        if declaration_only:
            return "%s * %s;" % (field_type, field_name)
        else:
            prefix = "" if no_declaration else "auto "
            return "%s%s = block->uncheckedFastGetData< %s >(%sID);" % (prefix, field_name, field_type, field_name)
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
def generate_field_parameters(ctx, kernel_info, parameters_to_ignore=[]):
    is_gpu = ctx['target'] == 'gpu'
    ast = kernel_info.ast
    fields = sorted(list(ast.fields_accessed), key=lambda f: f.name)
    field_accesses = ast.atoms(ResolvedFieldAccess)

    return ", ".join(["%s * %s" % (make_field_type(get_base_type(f.dtype),
                                                   get_field_fsize(f, field_accesses),
                                                   is_gpu), f.name)
                      for f in fields if f.name not in parameters_to_ignore])

@jinja2.contextfilter
def generate_block_data_to_field_extraction(ctx, kernel_info, parameters_to_ignore=[], parameters=None,
                                            declarations_only=False, no_declarations=False):
    ast = kernel_info.ast
    field_accesses = [a for a in ast.atoms(ResolvedFieldAccess) if a.field.name not in parameters_to_ignore]

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
def generate_call(ctx, kernel_info, ghost_layers_to_include=0, cell_interval=None, stream='0'):
    """Generates the function call to a pystencils kernel

    Args:
        kernel_info:
        ghost_layers_to_include: if left to 0, only the inner part of the ghost layer field is looped over
                                 a CHECK is inserted that the field has as many ghost layers as the pystencils AST
                                 needs. This parameter specifies how many ghost layers the kernel should view as
                                 "inner area". The ghost layer field has to have the required number of ghost layers
                                 remaining. Parameter has to be left to default if cell_interval is given.
        cell_interval: Defines the name (string) of a walberla CellInterval object in scope,
                       that defines the inner region for the kernel to loop over. Parameter has to be left to default
                       if ghost_layers_to_include is specified.
        stream: optional name of cuda stream variable
    """
    assert isinstance(ghost_layers_to_include, str) or ghost_layers_to_include >= 0
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

    def get_start_coordinates(parameter):
        field_object = fields[parameter.field_name]
        if cell_interval is None:
            return [-ghost_layers_to_include - required_ghost_layers] * field_object.spatial_dimensions
        else:
            assert ghost_layers_to_include == 0
            return [sp.Symbol("{ci}.{coord}Min()".format(coord=coord, ci=cell_interval)) - required_ghost_layers
                    for coord in ('x', 'y', 'z')]

    def get_end_coordinates(parameter):
        field_object = fields[parameter.field_name]
        if cell_interval is None:
            shape_names = ['xSize()', 'ySize()', 'zSize()'][:field_object.spatial_dimensions]
            offset = 2 * ghost_layers_to_include + 2 * required_ghost_layers
            return ["%s->%s + %s" % (parameter.field_name, e, offset) for e in shape_names]
        else:
            assert ghost_layers_to_include == 0
            coord_names = ['x', 'y', 'z'][:field_object.spatial_dimensions]
            return ["{ci}.{coord}Size() + {gl}".format(coord=coord, ci=cell_interval, gl=required_ghost_layers)
                    for coord in coord_names]

    for param in ast.parameters:
        if param.is_field_argument and FieldType.is_indexed(fields[param.field_name]):
            continue

        if param.is_field_ptr_argument:
            field = fields[param.field_name]
            if field.field_type == FieldType.BUFFER:
                kernel_call_lines.append("%s %s = %s;" % (param.dtype, param.name, param.field_name))
            else:
                coordinates = get_start_coordinates(param)
                actual_gls = "int_c(%s->nrOfGhostLayers())" % (param.field_name, )
                for c in set(coordinates):
                    kernel_call_lines.append("WALBERLA_ASSERT_GREATER_EQUAL(%s, -%s);" %
                                             (c, actual_gls))
                while len(coordinates) < 4:
                    coordinates.append(0)
                kernel_call_lines.append("%s %s = %s->dataAt(%s, %s, %s, %s);" %
                                         ((param.dtype, param.name, param.field_name) + tuple(coordinates)))

        elif param.is_field_stride_argument:
            type_str = get_base_type(param.dtype).base_name
            stride_names = ['xStride()', 'yStride()', 'zStride()', 'fStride()']
            stride_names = ["%s(%s->%s)" % (type_str, param.field_name, e) for e in stride_names]
            field = fields[param.field_name]
            strides = stride_names[:field.spatial_dimensions]
            if field.index_dimensions > 0:
                additional_strides = [1]
                for shape in reversed(field.index_shape[1:]):
                    additional_strides.append(additional_strides[-1] * shape)
                assert len(additional_strides) == field.index_dimensions
                f_stride_name = stride_names[-1]
                strides.extend(["%s(%d * %s)" % (type_str, e, f_stride_name) for e in reversed(additional_strides)])
            if is_cpu:
                kernel_call_lines.append("const %s %s [] = {%s};" % (type_str, param.name, ", ".join(strides)))
            else:
                kernel_call_lines.append("const %s %s_cpu [] = {%s};" % (type_str, param.name, ", ".join(strides)))
                kernel_call_lines.append(
                    "WALBERLA_CUDA_CHECK( cudaMemcpyToSymbolAsync(internal_%s::%s, %s_cpu, %d * sizeof(%s), 0, cudaMemcpyHostToDevice, %s) );"
                    % (ast.function_name, param.name, param.name, len(strides), type_str, stream))

        elif param.is_field_shape_argument:
            field = fields[param.field_name]
            type_str = get_base_type(param.dtype).base_name
            shapes = ["%s(%s)" % (type_str, c) for c in get_end_coordinates(param)]
            spatial_shape_symbols = [sp.Symbol("%s_cpu[%d]" % (param.name, i)) for i in range(field.spatial_dimensions)]

            max_values = ["%s->%sSizeWithGhostLayer()" % (field.name, coord) for coord in ['x', 'y', 'z']]
            for shape, max_value in zip(shapes, max_values):
                kernel_call_lines.append("WALBERLA_ASSERT_GREATER_EQUAL(%s, %s);" % (max_value, shape))

            if field.index_dimensions == 1:
                shapes.append("%s(%s->fSize())" % (type_str, field.name))
            elif field.index_dimensions > 1:
                shapes.extend(["%s(%d)" % (type_str, e) for e in field.index_shape])
                kernel_call_lines.append("WALBERLA_ASSERT_EQUAL(int(%s->fSize()),  %d);" %
                                         (field.name, prod(field.index_shape)))
            if is_cpu:
                kernel_call_lines.append("const %s %s [] = {%s};" % (type_str, param.name, ", ".join(shapes)))
            else:
                kernel_call_lines.append("const %s %s_cpu [] = {%s};" % (type_str, param.name, ", ".join(shapes)))
                kernel_call_lines.append(
                    "WALBERLA_CUDA_CHECK( cudaMemcpyToSymbolAsync(internal_%s::%s, %s_cpu, %d * sizeof(%s), 0, cudaMemcpyHostToDevice, %s) );"
                    % (ast.function_name, param.name, param.name, len(shapes), type_str, stream))

    if not is_cpu:
        indexing_dict = ast.indexing.call_parameters(spatial_shape_symbols)
        call_parameters = ", ".join([p.name for p in ast.parameters
                                     if p.is_field_ptr_argument or not p.is_field_argument])
        sp_printer_c = CustomSympyPrinter()

        kernel_call_lines += [
            "dim3 _block(int(%s), int(%s), int(%s));" % tuple(sp_printer_c.doprint(e) for e in indexing_dict['block']),
            "dim3 _grid(int(%s), int(%s), int(%s));" % tuple(sp_printer_c.doprint(e) for e in indexing_dict['grid']),
            "internal_%s::%s<<<_grid, _block, 0, %s>>>(%s);" % (ast.function_name, ast.function_name,
                                                                stream, call_parameters),
        ]
    else:
        kernel_call_lines.append("internal_%s::%s(%s);" %
                                 (ast.function_name, ast.function_name, ", ".join([p.name for p in ast.parameters])))
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
    varying_parameters = []
    if hasattr(kernel_info, 'varying_parameters'):
        varying_parameters = kernel_info.varying_parameters
    varying_parameter_names = [e[1] for e in varying_parameters]
    parameters_to_ignore += kernel_info.temporary_fields + varying_parameter_names

    parameter_list = []
    for param in ast.parameters:
        if param.is_field_ptr_argument and param.field_name not in parameters_to_ignore:
            parameter_list.append("BlockDataID %sID_" % (param.field_name, ))
        elif not param.is_field_argument and param.name not in parameters_to_ignore:
            parameter_list.append("%s %s_" % (param.dtype, param.name,))
    varying_parameters = ["%s %s_" % e for e in varying_parameters]
    return ", ".join(parameter_list + varying_parameters)


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
    jinja_env.filters['generate_field_parameters'] = generate_field_parameters
