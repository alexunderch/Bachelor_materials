#include "Common.h"

using namespace std;

// Этот файл сдаётся на проверку
// Здесь напишите реализацию необходимых классов-потомков `IShape`

class Polygon : public IShape {
public:

   void SetPosition(Point position) override { point = move(position); }
   Point GetPosition() const override { return point; }

   void SetSize(Size size) override { size_ = move(size); }
   Size GetSize() const override { return size_ ; }

   void SetTexture(shared_ptr<ITexture> texture) override { itexture_ = move(texture); }
   ITexture* GetTexture() const override {return itexture_.get(); }

   void Draw(Image& image) const override { return DrawFigure(image); }
   unique_ptr<IShape> Clone() const override { return CloneFigure(); }

protected:
    Point point;
    Size size_;
    shared_ptr<ITexture> itexture_;
    virtual void DrawFigure(Image& image) const = 0;
    virtual unique_ptr<IShape> CloneFigure () const = 0;
};

class Rectangle : public Polygon {
    void DrawFigure(Image& image) const override {
         for (int y = point.y; y < point.y + size_.height ; ++y) {
            for (int x = point.x; x < point.x + size_.width ; ++x) {
                if (y < image.size() && x < image.begin() -> size()) {
                    if (GetTexture() && y - point.y < itexture_ -> GetSize().height &&
                        x - point.x < itexture_ -> GetSize().width) {
                        image[y][x] = GetTexture() -> GetImage()[y - point.y][x - point.x];        
                    }
                    else image[y][x] = '.';
                }
            }
        }
    }

    unique_ptr<IShape> CloneFigure () const {
        Rectangle rectangle;
        rectangle.SetPosition(point);
        rectangle.SetSize(size_);
        rectangle.SetTexture(itexture_);
        return make_unique<Rectangle>(rectangle);
    }

};

class Ellipse : public Polygon {
    void DrawFigure(Image& image) const override {
         for (int y = point.y; y < point.y + size_.height ; ++y) {
            for (int x = point.x; x < point.x + size_.width ; ++x) {
                if (IsPointInEllipse({x - point.x, y - point.y}, size_)) {
                    if (y < image.size() && x < image.begin() -> size()) {
                        if (GetTexture() && y - point.y < itexture_ ->GetSize().height 
                            && x - point.x < itexture_-> GetSize().width) {
                            image[y][x] = GetTexture() -> GetImage()[y - point.y][x - point.x];        
                        }
                        else image[y][x] = '.';
                    }
                }
            }
        }
    }

    unique_ptr<IShape> CloneFigure () const {
        Ellipse ellipse;
        ellipse.SetPosition(point);
        ellipse.SetSize(size_);
        ellipse.SetTexture(itexture_);
        return make_unique<Ellipse>(ellipse);
    }

};

// Напишите реализацию функции
unique_ptr<IShape> MakeShape(ShapeType shape_type) {
    if (shape_type == ShapeType::Ellipse) {
        return make_unique<Ellipse>();
    }
    if (shape_type == ShapeType::Rectangle) {
        return make_unique<Rectangle>();
    }
}

/*
// Точка передаётся в локальных координатах
bool IsPointInSize(Point p, Size s) {
  return p.x >= 0 && p.y >= 0 && p.x < s.width && p.y < s.height;
}

Size GetImageSize(const Image& image) {
  auto width = static_cast<int>(image.empty() ? 0 : image[0].size());
  auto height = static_cast<int>(image.size());
  return {width, height};
}

class Shape : public IShape {
public:
  void SetPosition(Point position) override {
    position_ = position;
  }
  Point GetPosition() const override {
    return position_;
  }

  void SetSize(Size size) override {
    size_ = size;
  }
  Size GetSize() const override {
    return size_;
  }

  void SetTexture(shared_ptr<ITexture> texture) override {
    texture_ = move(texture);
  }
  ITexture* GetTexture() const override {
    return texture_.get();
  }

  void Draw(Image& image) const override {
    Point p;
    auto image_size = GetImageSize(image);
    for (p.y = 0; p.y < size_.height; ++p.y) {
      for (p.x = 0; p.x < size_.width; ++p.x) {
        if (IsPointInShape(p)) {
          char pixel = '.';
          if (texture_ && IsPointInSize(p, texture_->GetSize())) {
            pixel = texture_->GetImage()[p.y][p.x];
          }
          Point dp = {position_.x + p.x, position_.y + p.y};
          if (IsPointInSize(dp, image_size)) {
            image[dp.y][dp.x] = pixel;
          }
        }
      }
    }
  }

private:
  // Вызывается только для точек в ограничивающем прямоугольнике
  // Точка передаётся в локальных координатах
  virtual bool IsPointInShape(Point) const = 0;

  shared_ptr<ITexture> texture_;
  Point position_ = {};
  Size size_ = {};
};

class Rectangle : public Shape {
public:
  unique_ptr<IShape> Clone() const override {
    return make_unique<Rectangle>(*this);
  }

private:
  bool IsPointInShape(Point) const override {
    return true;
  }
};

class Ellipse : public Shape {
public:
  unique_ptr<IShape> Clone() const override {
    return make_unique<Ellipse>(*this);
  }

private:
  bool IsPointInShape(Point p) const override {
    return IsPointInEllipse(p, GetSize());
  }
};

unique_ptr<IShape> MakeShape(ShapeType shape_type) {
  switch (shape_type) {
    case ShapeType::Rectangle:
      return make_unique<Rectangle>();
    case ShapeType::Ellipse:
      return make_unique<Ellipse>();
  }
  return nullptr;
}
*/