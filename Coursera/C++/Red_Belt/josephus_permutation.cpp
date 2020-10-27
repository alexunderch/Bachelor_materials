#include "test_runner.h"

#include <cstdint>
#include <iostream>
#include <cmath>
#include <iterator>
#include <numeric>
#include <list>
#include <vector>
#include <utility>

using namespace std;

template<typename RandomIt>
void MakeJosephusPermutation(RandomIt first, RandomIt last, uint32_t step_size)
{
	list<typename RandomIt::value_type> pool;
	move(first, last, back_inserter(pool));
  int cur_pos = 0;
  int pred_pos = 0;
  auto it = pool.begin();

      while (!pool.empty())
    {
      int tmp_pred = pred_pos, tmp_cur = cur_pos;
      if (cur_pos < pred_pos)
      {
        for (int ind = pred_pos; ind != cur_pos ; ind--)
        {
          if (prev(it) == pool.end()) {--it = prev(pool.end()); }
          else --it;
        }
      }
      else  {
        for (int ind = pred_pos; ind < cur_pos ; ind++)
        {
          if (next(it) == pool.end()) {++it = pool.begin(); }
          else ++it;
        }
      }

    //  cout << *it << endl;
      *(first++) = move(*it);
      if (next(it) != pool.end()) { pool.erase(it++);}
      else {it = pool.begin(); pool.pop_back();}

      if (pool.empty()) break;

      //pred_pos = tmp_pred; cur_pos = tmp_cur;
      pred_pos = cur_pos;
      cur_pos = (pred_pos + step_size - 1) % pool.size();
    //  cout << "p " << pred_pos << " c " << cur_pos << endl;
    //for (auto item: pool) {cout << item << " ";}
     //cout << endl;
    }

}

/* Author's solution
template <typename Container, typename ForwardIt>
ForwardIt LoopIterator(Container& container, ForwardIt pos) {
  return pos == container.end() ? container.begin() : pos;
}

template <typename RandomIt>
void MakeJosephusPermutation(RandomIt first, RandomIt last,
                             uint32_t step_size) {
  list<typename RandomIt::value_type> pool;
  for (auto it = first; it != last; ++it) {
    pool.push_back(move(*it));
  }
  auto cur_pos = pool.begin();
  while (!pool.empty()) {
    *(first++) = move(*cur_pos);
    if (pool.size() == 1) {
      break;
    }
    const auto next_pos = LoopIterator(pool, next(cur_pos));
    pool.erase(cur_pos);
    cur_pos = next_pos;
    for (uint32_t step_index = 1; step_index < step_size; ++step_index) {
      cur_pos = LoopIterator(pool, next(cur_pos));
    }
  }
}
*/
