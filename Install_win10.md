# install cond norm in win10

Follow these steps to install under win10.(test passed with win10, VS2017, pytorch 1.2)  

1. call Visual Studio environment, run command like this:  
```cmd
{VS_HOME}\2017\Community\VC\Auxiliary\Build\vcvars64.bat
set %INCLUDE%=.;%INCLUDE%
```

2. if use win10 with non english version, do this.  
open file `{PythonEnvPath}\lib\site-packages\torch\utils\cpp_extension.py`  
change decode scheme at line 185 under function `check_compiler_abi_compatibility`  
eg. 'gbk' in Chinese like:  
```python
match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode().strip())
match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(' gbk').strip())
```

3. modify some c++ header.  
file1: {PythonEnvPath}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h (line 181)  
`static constexpr size_t DEPTH_LIMIT = 128;`
    -->
`static const size_t DEPTH_LIMIT = 128;`  
file2: {PythonEnvPath}\Lib\site-packages\torch\include\pybind11\cast.h(line 1449)  
`explicit operator type&() { return *(this->value); }`
    -->
`explicit operator type&() { return *((type*)this->value); }`  


4. compile and install as ReadMe  
```cmd
cd ./xcn_cuda
python setup.py build
python setup.py install
```

