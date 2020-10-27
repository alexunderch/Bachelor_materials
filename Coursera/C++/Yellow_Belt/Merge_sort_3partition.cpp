#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace std;

template <typename RandomIt>
void MergeSort(RandomIt range_begin, RandomIt range_end)
{
  auto range_length = range_end - range_begin;
  if (range_length < 2) return;

  vector<typename RandomIt::value_type> elements(range_begin, range_end);
  vector<typename RandomIt::value_type> tmp;

  auto one_third = elements.begin() + range_length / 3;
  auto two_third = elements.begin() + 2 * range_length / 3;

  MergeSort(begin(elements), one_third);
  MergeSort(one_third, two_third);
  MergeSort(two_third, end(elements));

  merge(begin(elements), one_third, one_third, two_third, back_inserter(tmp));
  merge(tmp.begin(), tmp.end(), two_third, end(elements), range_begin);
}

int main() {
  vector<int> v = {6, 4, 7, 6, 4, 4, 0, 1, 5};
  MergeSort(begin(v), end(v));
  for (int x : v) {
    cout << x << " ";
  }
  cout << endl;
  return 0;
}
