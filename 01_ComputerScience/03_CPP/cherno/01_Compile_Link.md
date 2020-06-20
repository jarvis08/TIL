# Compile & Link

Reference: [The Cherno](https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb)

1. If we try to **compile cpp files**, **translation units** translate cpp files into **object files**. A translation unit is responsible for a cpp file.

2. Before compiling, **preprocessor statement** will be handled.

   ```cpp
   // INTEGER works as int
   #define INTEGER int
   
   // copy iostream.h and paste into this file
   #include <iostream>
   
   // compile code in condition
   #if 1
   ...
   #endif
   ```

3. If a function's code is defined in other cpp file(file-2) and if you want to use the function in the file-1, you need to write **decoration** into the file-1. If you use decoration, then **Linker** will link the files.

   ```cpp
   // main.cpp
   void say(const char* message); // decoration
   // don't have to mension 'message', but it looks better
   
   int main() {
   	say("print this");
   }
   ```

   ```cpp
   // say_something.cpp
   #include <iostream>
   
   void say(const char* message) {
     std::cout << message << std::endl;
   }
   ```

4. If project's size gets bigger, it will be more simple to include just **header file**. Header files usually only have decorations of functions, defined in seperate cpp file.

   ```cpp
   // main.cpp
   #include "say_something.h"
   ```

   

   

   

   