"""
Mechanism for integrating code generation into waLBerla's CMake system.
CMake needs to determine which C++ source files are generated by which Python script.
The list of files should be available fast, without running the generation process itself.
Thus all code generation function are registered at a single class that manages this process.

Usage example:
    from pystencils_walberla.cmake_integration import codegen
    codegen.register(['MyClass.h', 'MyClass.cpp'], functionReturningTwoStringsForHeaderAndCpp)

"""
import atexit
from argparse import ArgumentParser


class CodeGeneratorCMakeIntegration:

    def __init__(self):
        self._registeredGenerators = []

    def register(self, files, generation_function):
        """
        Register function that generates on or more source files
        :param files: paths of files to generate
        :param generation_function: function that returns a tuple of string with the file contents
                                   returned tuple has to have as many entries as files
        """
        self._registeredGenerators.append((files, generation_function))

    @property
    def generated_files(self):
        return sum((e[0] for e in self._registeredGenerators), [])

    def generate(self):
        for paths, generatorFunction in self._registeredGenerators:
            files = generatorFunction()
            assert len(files) == len(paths), "Registered generator function does not return expected amount of files"
            for path, file in zip(paths, files):
                with open(path, 'w') as f:
                    f.write(file)


codegen = CodeGeneratorCMakeIntegration()


def main():
    from pystencils.gpucuda.indexing import AUTO_BLOCK_SIZE_LIMITING

    # prevent automatic block size detection of CUDA generation module
    # this would import pycuda, which might not be available, and if it is available problems occur
    # since we use atexit and pycuda does as well, leading to a tear-down problem
    previous_block_size_limiting_state = AUTO_BLOCK_SIZE_LIMITING
    AUTO_BLOCKSIZE_LIMITING = False

    parser = ArgumentParser()
    parser.add_argument("-l", "--list-output-files", action='store_true', default=False,
                        help="Prints a list of files this script generates instead of generating them")
    parser.add_argument("-g", "--generate", action='store_true', default=False,
                        help="Generates the files")

    args = parser.parse_args()
    if args.list_output_files:
        print(";".join(codegen.generated_files))
    elif args.generate:
        codegen.generate()
    else:
        parser.print_help()
    AUTO_BLOCKSIZE_LIMITING = previous_block_size_limiting_state


atexit.register(main)
