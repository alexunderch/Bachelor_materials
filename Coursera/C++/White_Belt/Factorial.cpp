#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
using namespace std;

int Factorial(int);

int main(void)
{
	int x = 0;
	cin >> x;
	cout << Factorial(x) << endl;
	return 0;
}

int Factorial (int x)
{
	if (x < 2) 
	{
		return 1;
	}
	
	else 
	return Factorial(x-1)*x;
}


