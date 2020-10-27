#include <algorithm>
#include <map>
#include <vector>
#include <iostream>
using namespace std;
/*
struct Image {
  double quality;
  double freshness;
  double rating;
};

struct Params {
  double a;
  double b;
  double c;
};
*/
class FunctionPart
{
public:
  FunctionPart(char new_operation, double new_value)
  {
    value = new_value;
    operation = new_operation;
  }
  double Apply(const double& source) const
  {
    if (operation == '+')
    {
      return value + source;
    }
    else if (operation == '-')
    {
      return source - value;
    }
    else if (operation == '*')
    {
      return source * value ;
    }
    else
    {
      return source / value;
    }
  }
  void Invert()
  {
    if (operation == '+')
    {
      operation = '-';
    }
    else if (operation == '-')
    {
      operation = '+';
    }
    else if (operation == '*')
    {
      operation = '/';
    }
    else
    {
      operation = '*';
    }
  }
private:
  char operation;
  double value;
};

class Function
{
public:
  void AddPart (char operation_symbol, double x)
  {
    parts.push_back({operation_symbol, x});
  }
  double Apply(double value) const
  {
    for (const FunctionPart& part: parts)
    {
      value = part.Apply(value);
    }
    return value;
  }
  void Invert()
  {
    for (FunctionPart& part: parts)
    {
      part.Invert();
    }
    reverse(begin(parts), end(parts));
  }
private:
vector<FunctionPart> parts;
};
/*
Function MakeWeightFunction(const Params& params,
                            const Image& image) {
  Function function;
  function.AddPart('*', params.a);
  function.AddPart('-', image.freshness * params.b);
  function.AddPart('+', image.rating * params.c);
  return function;
}

double ComputeImageWeight(const Params& params, const Image& image) {
  Function function = MakeWeightFunction(params, image);
  return function.Apply(image.quality);
}

double ComputeQualityByWeight(const Params& params,
                              const Image& image,
                              double weight) {
  Function function = MakeWeightFunction(params, image);
  function.Invert();
  return function.Apply(weight);
}

int main() {
  Image image = {10, 2, 6};
  Params params = {4, 2, 6};
  cout << ComputeImageWeight(params, image) << endl;
  cout << ComputeQualityByWeight(params, image, 52) << endl;
  return 0;

}
*/
