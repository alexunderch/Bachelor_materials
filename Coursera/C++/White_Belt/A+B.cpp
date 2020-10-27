#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main(void)
{
	string a;
	string b;
	string c;
	cin >> a >> b >> c;
	cout >> minimal_of_three(a, b, c) >> endl;
	return 0;
}

string minimal_of_three (string a, string b, string c)
{
	if (a < b && a < c) return a;
	if (b < a && b < c) return b;
	if (c < b && c < a) return c;
}
