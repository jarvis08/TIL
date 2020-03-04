#  Compile without IDE

IDE, Integrated Development Environments는 알아서 컴파일 후 실행해 주지만, text editor를 사용해서 개발한다면 개발자가 따로 compile 작업을 한 후 실행할 수 있습니다.

1. gcc compiler 설치
2. `make` 명령어 만으로 header(`.h`), implementation(`.cpp`) 파일들을 한번에 컴파일 하도록 환경 구성
   - 이 강의에서는 `generic.mk` 파일을 생성하여 로드하도록 구성되어 있으며, `generic.mk` 파일과 로드하는 `Makefile` 파일은 아래 Appendix에서 정의
   - `generic.mk` 파일은 공통적으로 사용될 수 있기에 root에 위치
   - `Makefile` 파일은 프로젝트 마다 존재
3. `make` 명령어로 컴파일
4. `./compiled_filename` 명령어로 실행
   - `./` 상대 경로 없으면 에러
5. `make clean` 명령어를 통해 object 파일들과 executable file을 삭제

<br><br>

## 1. Appendix

### 1-1. `generic.mk` 내용

```makefile
## generic.mk

#
# This is a generic Makefile designed to compile a sample directory of code.
# This file depends on variables having been set before calling:
#   EXE: The name of the result file
#   OBJS: Array of objects files (.o) to be generated
#   CLEAN_RM: Optional list of additional files to delete on `make clean`
#
# @author Wade Fagen-Ulmschneider, <waf@illinois.edu>
# @author Jeffrey Tolar
# @author Eric Huber (edits made for CS Fundamentals MOOC)
#


# Compiler/linker config and object/depfile directory:
CXX = g++
LD  = g++
OBJS_DIR = .objs

# -MMD and -MP asks clang++ to generate a .d file listing the headers used in the source code for use in the Make process.
#   -MMD: "Write a depfile containing user headers"
#   -MP : "Create phony target for each dependency (other than main file)"
#   (https://clang.llvm.org/docs/ClangCommandLineReference.html)
DEPFILE_FLAGS = -MMD -MP

# Provide lots of helpful warning/errors:
# (Switching from clang++ to g++ caused some trouble here. Not all flags are identically between the compilers.)
#WARNINGS_AS_ERRORS = -Werror # Un-commenting this line makes compilation much more strict.
GCC_EXCLUSIVE_WARNING_OPTIONS =  # -Wno-unused-but-set-variable
CLANG_EXCLUSIVE_WARNING_OPTIONS =  # -Wno-unused-parameter -Wno-unused-variable
ifeq ($(CXX),g++)
EXCLUSIVE_WARNING_OPTIONS = $(GCC_EXCLUSIVE_WARNING_OPTIONS)
else
EXCLUSIVE_WARNING_OPTIONS = $(CLANG_EXCLUSIVE_WARNING_OPTIONS)
endif
# ASANFLAGS = -fsanitize=address -fno-omit-frame-pointer # for debugging, if supported on the OS
WARNINGS = -pedantic -Wall $(WARNINGS_AS_ERRORS) -Wfatal-errors -Wextra $(EXCLUSIVE_WARNING_OPTIONS)

# Flags for compile:
CXXFLAGS += -std=c++14 -O0 $(WARNINGS) $(DEPFILE_FLAGS) -g -c $(ASANFLAGS)

# Flags for linking:
LDFLAGS += -std=c++14 $(ASANFLAGS)

# Rule for `all` (first/default rule):
all: $(EXE)

# Rule for linking the final executable:
# - $(EXE) depends on all object files in $(OBJS)
# - `patsubst` function adds the directory name $(OBJS_DIR) before every object file
$(EXE): $(patsubst %.o, $(OBJS_DIR)/%.o, $(OBJS))
	$(LD) $^ $(LDFLAGS) -o $@

# Ensure .objs/ exists:
$(OBJS_DIR):
	@mkdir -p $(OBJS_DIR)
	@mkdir -p $(OBJS_DIR)/uiuc

# Rules for compiling source code.
# - Every object file is required by $(EXE)
# - Generates the rule requiring the .cpp file of the same name
$(OBJS_DIR)/%.o: %.cpp | $(OBJS_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Additional dependencies for object files are included in the clang++
# generated .d files (from $(DEPFILE_FLAGS)):
-include $(OBJS_DIR)/*.d
-include $(OBJS_DIR)/uiuc/*.d


# Standard C++ Makefile rules:
clean:
	rm -rf $(EXE) $(TEST) $(OBJS_DIR) $(CLEAN_RM) *.o *.d

tidy: clean
	rm -rf doc

.PHONY: all tidy clean
```

<br>

### 1-2. `Makefile` 내용

아래 파일을 프로젝트 내부에 생성 후, `make` 명령어 만으로도 다음 파일 구성을 컴파일

- `Cube.h`
- `Cube.cpp`
- `main.cpp`

```
EXE = main
OBJS = main.o Cube.o
CLEAN_RM =

include PATH/to/generic.mk
```

