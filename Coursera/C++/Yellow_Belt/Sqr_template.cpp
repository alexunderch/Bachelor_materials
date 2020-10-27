#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <sstream>

using namespace std;

template <typename T> T Sqr (const T&);

template <typename T> vector<T> operator * (const vector<T>&, const vector<T>&);
template <typename F, typename S> pair<F, S> operator * (const pair<F, S>&, const pair<F, S>&);
template <typename Key, typename Value> map<Key, Value> operator * (const map<Key, Value>&, const map<Key, Value>&);

// возведение в квадрат
template <typename T> T Sqr (const T& x)
{return x*x; }

template <typename T> vector<T> operator * (const vector<T>& v1)
{
//  if (v1.size() != v2.size()) return -1;
  std::vector<T> v;
  for (const auto& i: v1)
  {
    v.push_back(Sqr(i));
  }
  return v;
}

template <typename F, typename S> pair<F, S> operator * (const pair<F, S>& p1)
{
  return {Sqr(p1.first), Sqr(p1.second)};
}

template <typename Key, typename Value> map<Key, Value> operator * (const map<Key, Value>& m1)
{
  map<Key, Value> m;
  for (const auto& [key, value]: m1)
  {
    m[key] = Sqr(m1[key]);
  }
  return m;
}

int main()
{
vector<int> v = {1, 2, 3};
cout << "vector:";
for (int x : Sqr(v)) {
  cout << ' ' << x;
}
cout << endl;

map<int, pair<int, int>> map_of_pairs = {
  {4, {2, 2}},
  {7, {4, 3}}
};
cout << "map of pairs:" << endl;
for (const auto& x : Sqr(map_of_pairs)) {
  cout << x.first << ' ' << x.second.first << ' ' << x.second.second << endl;
}

  return 0;
}
