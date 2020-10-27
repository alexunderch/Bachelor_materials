#include <iostream>
#include <map>
#include <exception>

using namespace std;

template <typename Key, typename Value>
Value& GetRefStrict (map<Key, Value>& m, Key k)
{
  if (!m.count(k)) throw runtime_error("no such key in dictionary");
  return m.at(k);
}


int main()
{
map<int, string> m = {{0, "value"}};
string& item = GetRefStrict(m, 0);
item = "newvalue";
cout << m[0] << endl;
  return 0;
}
