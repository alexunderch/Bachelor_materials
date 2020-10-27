#include <iostream>
#include <vector>
using namespace std;

int main()
{
  int N = 0;
  cin >> N;
  vector<int> v(N);
  int temp = 0;
  int64_t avg = 0;

  for (size_t i = 0; i < v.size(); ++i)
  {
    cin >> temp;
    v[i] = temp;
    avg += temp;
  }

  avg /= static_cast<int>(v.size());

  vector<int> out;

  for (size_t i = 0; i < v.size(); ++i)
  {
    if (v[i] > avg)
    {
      out.push_back(i);
    }
  }

  cout << out.size() << endl;
  for (auto item: out)
  {
    cout << item << " ";
  }
  cout << endl;


  return 0;
}
