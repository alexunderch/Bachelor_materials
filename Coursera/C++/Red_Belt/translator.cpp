#include "test_runner.h"
#include <string>
#include <vector>
#include <map>
#include <string_view>
#include <deque>
#include <iostream>
using namespace std;

class Translator {
public:
  void Add(string_view source, string_view target)
  {
    const string_view source_view = GetClonedView(source);
    const string_view target_view = GetClonedView(target);
    forward_dict[source_view] = target_view;
    backward_dict[target_view] = source_view;
  }
  string_view TranslateForward(string_view source) const{
    return Translate(forward_dict, source);
  }
  string_view TranslateBackward(string_view target) const{
    return Translate(backward_dict, target);
  }

private:

  static string_view Translate(const map<string_view, string_view>& dict,
     string_view s) {
   if (const auto it = dict.find(s); it != dict.end()) {
     return it -> second;
   } else {
     return {};
   }
  }

  string_view GetClonedView(string_view s) {
  for (const auto* map_ptr : {&forward_dict, &backward_dict}) {
    const auto it = map_ptr -> find(s);
    if (it != map_ptr -> end()) {
      return it -> first;
    }
  }
  return data.emplace_back(s);
}

  map<string_view, string_view> forward_dict;
  map<string_view, string_view> backward_dict;
  deque<string> data;
};

void TestSimple() {
  Translator translator;
  translator.Add(string("okno"), string("window"));
  translator.Add(string("stol"), string("table"));

  ASSERT_EQUAL(translator.TranslateForward("okno"), "window");
  ASSERT_EQUAL(translator.TranslateBackward("table"), "stol");
  ASSERT_EQUAL(translator.TranslateBackward("stol"), "");
}

int main() {
  TestRunner tr;
  RUN_TEST(tr, TestSimple);
  return 0;
}
