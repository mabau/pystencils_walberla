from pystencils import Field, FieldType, Assignment, create_kernel, show_code
from jinja2 import Environment, PackageLoader
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env
from collections import namedtuple

KernelInfo = namedtuple("KernelInfo", ['ast', 'temporary_fields', 'field_swaps'])


field_to_pack = Field.create_generic("f", spatial_dimensions=3, index_dimensions=0)
buffer = Field.create_generic('buffer', spatial_dimensions=1, field_type=FieldType.BUFFER)

pack_eqs = [Assignment(buffer.center, field_to_pack.center)]

ast = create_kernel(pack_eqs, target='gpu')
ast.function_name = 'pack1'

env = Environment(loader=PackageLoader('pystencils_walberla'))
add_pystencils_filters_to_jinja_env(env)

context = {
    'kernel': KernelInfo(ast, [], []),
    'target': 'gpu',
}

header = env.get_template("Packing.tmpl.h").render(**context)
print(header)
