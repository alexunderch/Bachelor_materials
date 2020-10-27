#include <set>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

template <typename T>
vector<T> FindGreaterElements(const set<T>& elements, const T& border)
{
  auto iter = begin(elements);
  while(iter != end(elements) && *(iter) <= border)
  {
    ++iter;
  }
  return {iter, end(elements)};
}
/*
int main() {
  for (int x : FindGreaterElements(set<int>{1, 5, 7, 8}, 5)) {
    cout << x << " ";
  }
  cout << endl;

  string to_find = "Python";
  cout << FindGreaterElements(set<string>{"C", "C++"}, to_find).size() << endl;
  return 0;
}
*/
