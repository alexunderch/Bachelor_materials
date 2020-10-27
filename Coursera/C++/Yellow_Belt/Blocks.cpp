#include <iostream>
#include <cstdint>
using namespace std;

int main()
{
  int N = 0, R = 0;
  cin >> N >> R;
  uint64_t sum = 0;
  uint64_t vol = 0;
  int w = 0, h = 0, d = 0;

  while(N)
  {
    cin >> w >> h >> d;
    vol = R * w * h* d;
    sum += vol;
    --N;
  }
  cout << sum << endl;
}
