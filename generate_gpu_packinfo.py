import numpy as np
from pystencils import Field, FieldType, Assignment, create_kernel
from jinja2 import Environment, PackageLoader
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env
from collections import namedtuple

KernelInfo = namedtuple("KernelInfo", ['ast', 'temporary_fields', 'field_swaps'])
KernelSpec = namedtuple("KernelSpec", ["type", "layout", "dtype"])


def generate_gpu_packinfo():
    target = 'gpu'

    data_types = {
        'float': np.float32,
        'double': np.float64,
    }
    layouts = ['fzyx', 'zyxf']
    kernels = {}

    optimization = {'gpu_indexing': 'line'}

    for layout in layouts:
        for dtype_name, dtype in data_types.items():

            field_to_pack = Field.create_generic("f", spatial_dimensions=3, index_dimensions=0,
                                                 layout=layout, dtype=dtype)
            buffer = Field.create_generic('buffer', spatial_dimensions=1, field_type=FieldType.BUFFER, dtype=dtype)

            pack_ast = create_kernel([Assignment(buffer.center, field_to_pack.center)], target=target, **optimization)
            pack_ast.function_name = 'pack_%s_%s' % (dtype_name, layout)

            unpack_ast = create_kernel([Assignment(field_to_pack.center, buffer.center)], target=target, **optimization)
            unpack_ast.function_name = 'unpack_%s_%s' % (dtype_name, layout)

            kernels[KernelSpec('pack', layout, dtype_name)] = KernelInfo(pack_ast, [], [])
            kernels[KernelSpec('unpack', layout, dtype_name)] = KernelInfo(unpack_ast, [], [])

    env = Environment(loader=PackageLoader('pystencils_walberla'))
    add_pystencils_filters_to_jinja_env(env)

    context = {
        'layouts': layouts,
        'dtypes': list(data_types.keys()),
        'kernels': kernels,
        'target': target,
    }

    header = env.get_template("GpuPackInfo.tmpl.h").render(**context)
    source = env.get_template("GpuPackInfo.tmpl.cpp").render(**context)
    return header, source


if __name__ == '__main__':
    header, source = generate_gpu_packinfo()
    print(header, file=open('/local/bauer/code/walberla/tests/cuda/PackingTest.h', 'w'))
    print(source, file=open('/local/bauer/code/walberla/tests/cuda/PackingTest.cu', 'w'))
    print("Done")
