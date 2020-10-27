#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>

using namespace std;

//void Add(set< set<string> >&, const map<string, unsigned>& Syn_count,
							//	const string& word1, const string& word2);
//bool Check(const set< set<string> >&, 
								//const string& word1, const string& word2);


int main()
{
	int q;
	cin >> q;

	string command, word1, word2;
	map<string, set<string>> words;

	for (int i = 0; i < q; ++i)
	{
		cin >> command;

		if (command == "ADD")
		{
			cin >> word1 >> word2;

			words[word1].insert(word2);
			words[word2].insert(word1);
		}

		if (command == "COUNT")
		{
			cin >> word1;
			cout << words[word1].size() << endl;
		}

		if (command == "CHECK")
		{
			cin >> word1 >> word2;

			if (words[word1].count(word2) != 0)
				cout << "YES" << endl;
			else
				cout << "NO" << endl;
		}
	}
	
	return 0;
}
