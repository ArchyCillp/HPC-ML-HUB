---
title: string
---

基本用法
```C++
#include <iostream>
#include <string>

int main() {
    // Create a string
    std::string str = "Hello, World!";

    // Access characters
    char ch = str[0]; // 'H'
    ch = str.at(1);   // 'e'

    // Get string length
    size_t len = str.length(); // 13

    // Substring
    std::string substr = str.substr(7, 5); // "World"

    // Find substring
    size_t pos = str.find("World"); // 7

    // Replace substring
    str.replace(7, 5, "Universe"); // "Hello, Universe!"

    // Append to string
    str.append(" How are you?"); // "Hello, Universe! How are you?"

    // Insert into string
    str.insert(6, "Beautiful "); // "Hello, Beautiful Universe! How are you?"

    // Erase part of string
    str.erase(6, 10); // "Hello, Universe! How are you?"

    // Compare strings
    std::string str2 = "Hello, Universe! How are you?";
    bool isEqual = (str == str2); // true

    // Convert to C-style string
    const char* cstr = str.c_str();

    // Clear string
    str.clear(); // ""

    // Check if string is empty
    bool isEmpty = str.empty(); // true

    return 0;
}

