#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
using namespace std;

bool IsPalindrom(string);
vector<string> PalindromFilter(vector<string>, unsigned int);

int main(void)
{
	vector<string> s;
	string str ;
	char c;
	cin >> c;
	while (c != '\n')
	{
		cin >> str;
		s.push_back(str);
	}
	int L = 0;
	cin >> L;
	
	for(size_t i = 0; i < PalindromFilter(s, L).size(); ++i)
	{
        cout << PalindromFilter(s, L)[i] << ' ';
    }
	return 0;
}

bool IsPalindrom(string s)
{
	for (size_t i = 0; i < s.size()/2; ++i)
	{
		if (s[i] != s[s.size()-i-1])
		{
			return false;
		}
	}
	return true;
}

vector<string> PalindromFilter(vector<string> s, unsigned int minLength)
{
	vector<string> result;
	for (auto w: s)
	{
		if (IsPalindrom(w) && w.size() >= minLength)
		{
			result.push_back(w);
		}
	}
	return result;
}

