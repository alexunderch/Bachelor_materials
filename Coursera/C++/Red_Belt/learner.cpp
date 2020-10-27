#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <set>
#include "profile.h"
using namespace std;

class Learner {
 private:
  set<string> dict;
 public:
  int Learn(const vector<string>& words) {

    int newWords = 0;
    for (const auto& word : words) {

      if (dict.find(word) == end(dict))
      {
        ++newWords;
        dict.insert(word);
      }

    }
    return newWords;
  }

  vector<string> KnownWords() {
    return {dict.begin(), dict.end()};
  }
};

int main() {
  Learner learner;
  string line;
  while (getline(cin, line)) {

    LOG_DURATION("ADD ");
    vector<string> words;
    stringstream ss(line);
    string word;
    while (ss >> word) {
      words.push_back(word);
    }
    cout << learner.Learn(words) << "\n";
  }
  cout << "=== known words ===\n";
  for (auto word : learner.KnownWords()) {
    cout << word << "\n";
  }
}
