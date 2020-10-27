#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

void PrintVectorPart(const vector<int>& numbers)
{
  auto neg_it = find_if(begin(numbers), end(numbers),[](int x)
  {
      return x < 0;
  });
  for (auto iter = neg_it; iter != begin(numbers))
  {
    cout << *(--iter) << " ";
  }

}

int main() {
  PrintVectorPart({6, 1, 8, -5, 4});
  cout << endl;

  return 0;
}
