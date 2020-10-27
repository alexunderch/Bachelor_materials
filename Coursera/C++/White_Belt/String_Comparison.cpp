#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cctype>
#include <cmath>
using namespace std;

string ToLow(const string& S)
{
	string res;
    for (const auto& s: S)
		res += tolower(s);
	return res;
}


int main()
{

	int n = 0;
	cin >> n;
	string word;
	vector <string> words;
	while(n)
	{
		cin >> word;
		words.push_back(word);
		--n;
	}
	
	sort(begin(words), end(words), [](const string& x, const string& y)
	{
		return(ToLow(x) < ToLow(y));
	});
	for (auto& word: words)
	{
		cout << word << " ";
	}
	
	cout << endl;
	
	return 0;
}
