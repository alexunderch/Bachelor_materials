#include <string>
#include <iostream>
#include <list>
#include "test_runner.h"
using namespace std;

class Editor {
 public:
  // Реализуйте конструктор по умолчанию и объявленные методы
  Editor() : cursor_(text_.begin()) {text_.clear(); buffer.clear();}
  void Left() {
    if(text_.begin() == text_.end()) return;
    if (cursor_ == text_.begin()) {cursor_--; return;}
    cursor_--;
  }
  void Right() {
    if(text_.begin() == text_.end() || cursor_ == text_.end()) return;
    cursor_++;
  }
  void Insert(char token){
    text_.insert(cursor_, token);
  }
  void Cut(size_t tokens){
    auto tok = cursor_;
    cursor_ = NewOne(cursor_, tokens);
    buffer.clear();
    if (tokens == 0) return;
    buffer.splice(buffer.begin(), text_,  tok, cursor_);
    //cout << *cursor_ << " ";
  }
  void Copy(size_t tokens){
    auto tok = cursor_;
    cursor_ = NewOne(cursor_, tokens);
    buffer.clear();
    if (tokens == 0) return;
    for(auto it = tok; it != cursor_; it++) {
      buffer.push_back(*it);
    }
}
  void Paste() {
    if (buffer.empty()) return;
    //cout << GetText(buffer) << " ";
    for(auto it = buffer.begin(); it != buffer.end(); it++) {
      text_.insert(cursor_, *it);
    }
    //Right();
  }
  string GetText() const {
    string result;
    for(auto it = text_.begin(); it != text_.end(); it++) { result += *it; }
    return result;
  }
private:
  string GetText(const list<char>& t_) const {
    string result;
    for(auto it = t_.begin(); it != t_.end(); it++) { result += *it; }
    return result;
  }

  list<char>::iterator NewOne(list<char>::iterator& it, size_t tokens)
  {
    while (tokens != 0) {
      it++;
      if (it == text_.end()) {return text_.end();}
      tokens--;}
      return it;
  }

  list<char> text_;
  list<char> buffer;
  list<char>::iterator cursor_;
};

void TypeText(Editor& editor, const string& text) {
  for(char c : text) {
    editor.Insert(c);
  }
}

void TestEditing() {
  {
    Editor editor;

    const size_t text_len = 12;
    const size_t first_part_len = 7;
    TypeText(editor, "hello, world");

    for(size_t i = 0; i < text_len; ++i) {
      editor.Left();
    }

    editor.Cut(first_part_len);
    for(size_t i = 0; i < text_len - first_part_len; ++i) {
      editor.Right();
    }
    TypeText(editor, ", ");
    editor.Paste();

    editor.Left();
    editor.Left();
    editor.Cut(3);
    ASSERT_EQUAL(editor.GetText(), "world, hello");
  }
  {
    Editor editor;

    TypeText(editor, "misprnit");
    editor.Left();
    editor.Left();
    editor.Left();
    editor.Cut(1);
    editor.Right();
    editor.Paste();

    ASSERT_EQUAL(editor.GetText(), "misprint");
  }

}

void TestReverse() {
  Editor editor;

  const string text = "esreveR";
  for(char c : text) {
    editor.Insert(c);
    editor.Left();
  }

  ASSERT_EQUAL(editor.GetText(), "Reverse");
}

void TestNoText() {
  Editor editor;
  ASSERT_EQUAL(editor.GetText(), "");

  editor.Left();
  editor.Left();
  editor.Right();
  editor.Right();
  editor.Copy(0);
  editor.Cut(0);
  editor.Paste();

  ASSERT_EQUAL(editor.GetText(), "");
}

void TestEmptyBuffer() {
  Editor editor;

  editor.Paste();
  TypeText(editor, "example");
  editor.Left();
  editor.Left();
  editor.Paste();
  editor.Right();
  editor.Paste();
  editor.Copy(0);
  editor.Paste();
  editor.Left();
  editor.Cut(0);
  editor.Paste();

  ASSERT_EQUAL(editor.GetText(), "example");
}

void MyTest()
{
    Editor editor;

    const string text = "copy";
    const string paste_text = "pypy";
    TypeText(editor, text);

    editor.Right();

    for (size_t i = 0; i < text.size(); i++)
        editor.Left();

    editor.Cut(2);
    editor.Cut(3);
    editor.Paste();
    editor.Paste();

    ASSERT_EQUAL(editor.GetText(), paste_text);
}

void MyTest_2 ()
{
  Editor editor;

TypeText(editor, "1234567");

  editor.Left();
  editor.Left();
  editor.Cut(1);
  editor.Paste();
  editor.Paste();
  editor.Paste();

  ASSERT_EQUAL(editor.GetText(), "123456667");
}


int main() {
  TestRunner tr;
  RUN_TEST(tr, MyTest);
  RUN_TEST(tr, MyTest_2);
  RUN_TEST(tr, TestEditing);
  RUN_TEST(tr, TestReverse);
  RUN_TEST(tr, TestNoText);
  RUN_TEST(tr, TestEmptyBuffer);
  return 0;
}
/* АВТОРСКОЕ РЕШЕНИЕ:
#include <list>
#include <string>

using namespace std;

class Editor {
public:
  Editor()
    : pos(text.end()) {
  }

  void Left() {
    pos = Advance(pos, -1);
  }

  void Right() {
    pos = Advance(pos, 1);
  }

  void Insert(char token) {
    text.insert(pos, token);
  }

  void Cut(size_t tokens = 1) {
    auto pos2 = Advance(pos, tokens);
    buffer.assign(pos, pos2);
    pos = text.erase(pos, pos2);
  }

  void Copy(size_t tokens = 1) {
    auto pos2 = Advance(pos, tokens);
    buffer.assign(pos, pos2);
  }

  void Paste() {
    text.insert(pos, buffer.begin(), buffer.end());
  }

  string GetText() const {
    return {text.begin(), text.end()};
  }

private:
  using Iterator = list<char>::iterator;
  list<char> text;
  list<char> buffer;
  Iterator pos;

  Iterator Advance(Iterator it, int steps) const {
    while (steps > 0 && it != text.end()) {
      ++it;
      --steps;
    }
    while (steps < 0 && it != text.begin()) {
      --it;
      ++steps;
    }
    return it;
  }
};

*/
