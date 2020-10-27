#include <algorithm>
#include <iostream>
#include <string>
#include <string_view>
#include <set>
#include <sstream>
#include "test_runner.h"

using namespace std;

vector<string_view> SplitBy(string_view s, char sep = '.') {
  vector<string_view> result;
  while (!s.empty()) {
    size_t pos = s.find(sep);
    result.push_back(s.substr(0, pos));
    s.remove_prefix(pos != s.npos ? pos + 1 : s.size());
  }
  return result;
}

bool NecessaryBannedDomains(set<string>& banned_domains, string domain) {
  if (banned_domains.empty()) return true;
  vector<string_view> tmp = SplitBy(domain);
  string sum;
  for (size_t i = 0; i < tmp.size(); ++i) {
    if (i == 0) sum = static_cast<string>(tmp[0]);
    else sum = sum + "." + static_cast<string>(tmp[i]);
    if (banned_domains.count(sum)) return false;
  }
  return true;
}


set<string> ReadDomains(istream& is) {
 size_t count;
	is >> count;

	is.ignore();

	set<string> domains;
	for (size_t i = 0; i < count; ++i) {
		string domain;
		getline(is, domain);
    reverse(domain.begin(), domain.end());
    if (NecessaryBannedDomains(domains, domain)) domains.insert(domain);
	}
	return domains;
}

// void TestReadDomains () {
//   istringstream ss("4 ya.ru\nmaps.me\nm.ya.ru\ncom");
//   set<string> expected = {"com", "m.ya.ru", "maps.me", "ya.ru"};
//   ASSERT_EQUAL(ReadDomains(ss), expected ); 
// }

// void TestIsDomain () { 
//   ASSERT(IsSubdomain("ya.ru", "m.ya.ru")); 
//   ASSERT(IsSubdomain("ya.ru", "top.m.ya.ru")); 
//   ASSERT(IsSubdomain("ya.ru", "ru")); 
//   ASSERT(IsSubdomain("com.ru", "ru")); 
//   ASSERT(IsSubdomain("ru.com", "com"));
//   ASSERT(IsSubdomain("ya.com.ru", "com.ru"));
//   ASSERT(IsSubdomain("ya.com.ru", "ru"));   
// }

// void TestDomainsPreparation () {
//   set<string> v = {"google.co.uk", "m.ya.ya", "ya.ru", "m.sweet.con", "con", "ya.ya", "co.uk"};
//   set<string> expected = {"ya.ru", "con", "ya.ya", "co.uk"};
//   PrepareBannedDomains(v);
//   ASSERT_EQUAL(v, expected);
// }

int main() {
  TestRunner tr;
  //RUN_TEST(tr, TestReadDomains);
  //RUN_TEST(tr, TestIsDomain);
  //RUN_TEST(tr, TestDomainsPreparation);


  set<string> banned_domains = ReadDomains(cin);
  

  std::string input;
  size_t count;
  getline(cin, input);
  count = stoi(input);

  //vector<string> domains;
  for (size_t i = 0; i < count; ++i) {
    string domain;
    cin >> domain;
    reverse(domain.begin(), domain.end());
    if (NecessaryBannedDomains(banned_domains, domain)) {
        cout << "Good" << endl;
    } else {
        cout << "Bad" << endl;
    }
  }
  // set<string> domains_to_check = ReadDomains(cin);

  
  // for (const string domain : domains_to_check) {
  //   auto it = domains_to_check.upper_bound(domain);
	//   if (it != domains_to_check.begin() && IsSubdomain(domain, *prev(it))) {
  //   cout << "Good" << endl;
  //   } else {
  //     cout << "Bad" << endl;
  //   }
  // }
  return 0;
}

