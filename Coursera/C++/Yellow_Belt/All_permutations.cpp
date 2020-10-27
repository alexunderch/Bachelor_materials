//iota !!!!!!!!!!
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
void PrintVector(const vector<int>& v)
{
  for(auto it = begin(v); it < end(v); ++it)
    cout << *it << " ";
  cout << endl;
}
int main()
{
  int n = 0;
  cin >> n;
  vector<int> v;

  for (int i = n; i > 0; --i)
    v.push_back(i);

  do {
      PrintVector(v);
    } while(prev_permutation(v.begin(), v.end()));

  return 0;
}

/*
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

int main() {
  int range_length;
  cin >> range_length;
  vector<int> permutation(range_length);

  // iota             -> http://ru.cppreference.com/w/cpp/algorithm/iota
  // Заполняет диапазон последовательно возрастающими значениями
  iota(permutation.begin(), permutation.end(), 1);

  // reverse          -> http://ru.cppreference.com/w/cpp/algorithm/reverse
  // Меняет порядок следования элементов в диапазоне на противоположный
  reverse(permutation.begin(), permutation.end());

  // prev_permutation ->
  //     http://ru.cppreference.com/w/cpp/algorithm/prev_permutation
  // Преобразует диапазон в предыдущую (лексикографически) перестановку,
  // если она существует, и возвращает true,
  // иначе (если не существует) - в последнюю (наибольшую) и возвращает false
  do {
    for (int num : permutation) {
      cout << num << ' ';
    }
    cout << endl;
  } while (prev_permutation(permutation.begin(), permutation.end()));

  return 0;
}
*/
