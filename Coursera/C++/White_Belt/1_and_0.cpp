#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
using namespace std;

int main(void)
{
	int x = 0;
	cin >> x;
	string s;
	while (x)
	{
		
		if (x % 2) s += "1" ;
		else s += "0";
		
		x /= 2;
	}
	for (int i = s.size() - 1; i >= 0; --i) cout << s[i];
	
	cout << endl;
	return 0;
}

/*Через вектор:
 * #include <iostream>
#include <vector>
using namespace std;

int main() {
    int n;
    vector<int> bits;

    cin >> n;
    while (n > 0) {
        bits.push_back(n % 2);
        n /= 2;
    }

    for (int i = bits.size() - 1; i >= 0; --i) {
        cout << bits[i];
    }
    return 0;
}

 * */

