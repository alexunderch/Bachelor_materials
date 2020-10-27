#include <iostream>
#include <map>
#include <list>
#include <algorithm>

using namespace std;

int main()
{
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);

  using Position = list<int>::iterator;
  int number = 0;
  cin >> number;

  list<int> row;
  map<int, Position> pos;

  for (int i = 0; i < number; ++i)
  {
    int sp = 0, next_sp = 0;
    cin >> sp >> next_sp;
    if (!pos.count(next_sp))
    {
      row.push_back(sp);
      pos[sp] = prev(row.end());
    }
    else
    {
      Position nxt = pos[next_sp];
      row.insert(nxt, sp);
      pos[sp] = prev(nxt);
    }
  }

  for(auto x: row) { cout << x << '\n'; }
  return 0;
}
