#include "test_runner.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <iterator>
#include <vector>

using namespace std;

template <typename RandomIt>
void MergeSort(RandomIt range_begin, RandomIt range_end) {

  if (range_end - range_begin < 2) return;

  vector<typename RandomIt::value_type> elements;
  move(range_begin, range_end, back_inserter(elements));

  vector<typename RandomIt::value_type> tmp;
  //move()
  tmp.reserve(elements.size() + tmp.size());

  auto one_third = move(elements.begin() + (range_end - range_begin) / 3);
  auto two_third = move(elements.begin() + 2 * (range_end - range_begin) /3);

  MergeSort(elements.begin(), one_third);
  MergeSort(one_third, two_third);
  MergeSort(two_third, elements.end());

  merge(make_move_iterator(elements.begin()),
               make_move_iterator(one_third),
               make_move_iterator(one_third),
               make_move_iterator(two_third),
                         back_inserter(tmp));
                         
  merge(make_move_iterator(tmp.begin()),
          make_move_iterator(tmp.end()),
          make_move_iterator(two_third),
     make_move_iterator(elements.end()),
                           range_begin);
}

struct NoncopyableInt {
  int value;

  NoncopyableInt(const NoncopyableInt&) = delete;
  NoncopyableInt& operator=(const NoncopyableInt&) = delete;

  NoncopyableInt(NoncopyableInt&&) = default;
  NoncopyableInt& operator=(NoncopyableInt&&) = default;
};

bool operator == (const NoncopyableInt& lhs, const NoncopyableInt& rhs) {
  return lhs.value == rhs.value;
}

bool operator < (const NoncopyableInt& lhs, const NoncopyableInt& rhs) {
  return lhs.value < rhs.value;
}

ostream& operator << (ostream& os, const NoncopyableInt& v) {
  return os << v.value;
}

void TestIntVector() {
  /*
  vector<NoncopyableInt> numbers;
  numbers.push_back({6});
  numbers.push_back({1});
  numbers.push_back({3});
  numbers.push_back({9});
  numbers.push_back({1});
  numbers.push_back({9});
  numbers.push_back({9});
  numbers.push_back({8});
  numbers.push_back({12});
  numbers.push_back({1});
*/
  vector<int> numbers = {6, 1, 3, 9, 1, 9, 8, 12, 1};
  MergeSort(begin(numbers), end(numbers));
  //ASSERT(is_sorted(begin(numbers), end(numbers)));
}

int main() {
  TestRunner tr;
  RUN_TEST(tr, TestIntVector);
  return 0;
}
