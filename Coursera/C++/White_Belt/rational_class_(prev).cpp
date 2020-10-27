#include <iostream>
#include <exception>
using namespace std;

int LGD (int a, int b) {
  if (b == 0) {
    return a;
  } else {
    return LGD(b, a % b);
  }
}


int Abs(int& a)
{
  if (a < 0) return -a;
  return a;
}


class Rational {
public:
    Rational() {
        // Реализуйте конструктор по умолчанию
        p = 0;
        q = 1;
    }
    Rational(int numerator, int denominator) {
        // Реализуйте конструктор
        int sign = 1;

        if (numerator == 0) denominator = 1;
        if (denominator == 0)
        {
          throw invalid_argument("Invalid argument");
        }
        if ((numerator < 0 && denominator > 0) ||
           (numerator > 0 && denominator < 0))
        {
          sign = -1;
        }
        numerator = sign * Abs(numerator);
        denominator = Abs(denominator);

        int lgd = LGD (Abs(numerator), Abs(denominator));
        p = numerator / lgd;
        q = denominator / lgd;
    }
    int Numerator() const {
        // Реализуйте этот метод
        return p;
    }
    int Denominator() const {
        // Реализуйте этот метод
        return q;
    }
private:
    // Добавьте поля
    int p;
    int q;
};


// Реализуйте для класса Rational операторы ==, + и -
bool operator == (const Rational& l, const Rational& r)
{
  if (l.Numerator() == r.Numerator() && l.Denominator() == r.Denominator())
  {
    return true;
  }
  return false;
}
Rational operator + (Rational a, Rational b)
{
    int num  = (a.Numerator() * b.Denominator()) + (b.Numerator() * a.Denominator());
    int denom = a.Denominator() * b.Denominator();
    return Rational(num, denom);
}
Rational operator - (Rational a, Rational b)
{
    int num = (a.Numerator() * b.Denominator()) - (b.Numerator() * a.Denominator());
    int denom = a.Denominator() * b.Denominator();
    return Rational(num, denom);
}
// Реализуйте для класса Rational операторы * и /
Rational operator * (Rational a, Rational b)
{
  return Rational(a.Numerator() * b.Numerator(), a.Denominator() * b. Denominator());
}

Rational operator / (Rational a, Rational b)
{
  if (b.Numerator() == 0)
  {
    throw domain_error("Division by zero");
  }
  return Rational(a.Numerator() * b.Denominator(), a.Denominator() * b. Numerator());
}

// Реализуйте для класса Rational операторы << и >>
istream& operator >> (istream& stream, Rational& r)
{
  int p, q;
  if (stream >> p && stream.ignore(1) && stream >> q)
  {
    r = Rational(p, q);
  }
  return stream;
}

ostream& operator << (ostream& stream, const Rational& r)
{
  stream << r.Numerator() << "/" << r.Denominator();
  return stream;
}

// Реализуйте для класса Rational оператор(ы), необходимые для использования его
// в качестве ключа map'а и элемента set'а

bool operator > (Rational lhs, Rational rhs)
{
  if (lhs.Denominator() == rhs.Denominator())
  {
    return (lhs.Numerator() > rhs.Numerator());
  }
  else
  {
    return (lhs.Numerator() * rhs.Denominator()) > (rhs.Numerator() * lhs.Denominator());
  }
}
bool operator < (Rational lhs, Rational rhs)
{
  if (lhs.Denominator() == rhs.Denominator())
  {
    return (lhs.Numerator() < rhs.Numerator());
  }
  else
  {
    return (lhs.Numerator() * rhs.Denominator()) < (rhs.Numerator() * lhs.Denominator());
  }
}


int main()
 {
   try
   {
     Rational lhs, rhs;
     char sep;
     cin >> lhs >> sep >> rhs;
     switch (sep)
     {
       case '+':
       {
        cout << lhs + rhs << endl;
        break;
       }
       case '-':
       {
         cout << lhs - rhs << endl;
         break;
       }
       case '*':
       {
         cout << lhs * rhs << endl;
         break;
       }
       case '/':
       {
         cout << lhs/ rhs << endl;
         break;
       }
     }

   } catch (exception& ex)
   {
     cout << ex.what() << endl;
   }
    return 0;
}
