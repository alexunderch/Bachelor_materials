#include <iostream>
#include <string>
using namespace std;

class Animal {
public:
    Animal(const string& n = "animal") : Name(n) {}
    const string Name;
};


class Dog : public Animal {
public:
    Dog(const string& name) : Animal(name) {}
    void Bark() {
        cout << Name << " barks: woof!" << endl;
    }
};
