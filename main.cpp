#include <iostream>
#include <vector>
#include <memory>

#include <random>

using f64 = double;
using f32 = float;

static constexpr f64 Infinity = std::numeric_limits<f64>::infinity();
static constexpr f64 Pi = 3.1415926535897932385;


inline f64 DegToRad(f64 deg) {
    return (deg * Pi) / 180;
}

inline f64 RandomF64() {
    static std::uniform_real_distribution<f64> dist(0.0, 1.0);
    static std::mt19937 gen;

    return dist(gen);
}

inline f64 RandomF64(f64 min, f64 max) {
    return min + (max - min)*RandomF64();
}


class Vec3 {
public:
    f64 x, y, z;

    Vec3(): x{0},y{0},z{0} {}

    Vec3(f64 x, f64 y, f64 z): x{x}, y{y}, z{z} {}

    f64 Length() const {
        return sqrt(LengthSq());
    }

    f64 LengthSq() const {
        f64 res = x*x + y*y + z*z;

        return res;
    }

    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    Vec3& operator+=(const Vec3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        
        return *this;
    }

    bool NearZero() const {
        // return true if the vector is close to zero in all dimensions
        auto s = 1e-8;
        return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
    }

    static Vec3 Random() {
        return Vec3(RandomF64(), RandomF64(), RandomF64());
    }

    static Vec3 Random(f64 min, f64 max) {
        return Vec3(RandomF64(min, max), RandomF64(min, max), RandomF64(min, max));
    }
};

inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
    Vec3 res = {a.x + b.x, a.y + b.y, a.z + b.z};

    return res;
}

inline Vec3 operator*(f64 a, const Vec3 &b) {
  Vec3 res = {a * b.x, a * b.y, a * b.z};

  return res;
}

inline Vec3 operator*(const Vec3 &a, f64 b) {
  Vec3 res = {a.x * b, a.y * b, a.z * b};

  return res;
}

inline Vec3 operator*(const Vec3 &a, const Vec3 &b) {
  Vec3 res = {a.x * b.x, a.y * b.y, a.z * b.z};

  return res;
}

inline Vec3 operator/(const Vec3 &a, f64 b) {
    Vec3 res = {a.x / b, a.y / b, a.z / b};

    return res;
}

inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
    Vec3 res = {a.x - b.x, a.y - b.y, a.z - b.z};

    return res;
}

inline Vec3 Normalize(const Vec3 &v) {
    return v / v.Length();
}

inline f64 Dot(Vec3 a, Vec3 b) {
    f64 result = a.x * b.x + a.y * b.y + a.z * b.z;

    return result;
}

inline Vec3 Cross(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}

inline Vec3 RandomInUnitSphere() {
    for (;;) {
        Vec3 p = Vec3::Random(-1, 1);
        if (p.LengthSq() < 1)
            return p;
    }
}

inline Vec3 RandomNormalizedVector() {
    Vec3 v = RandomInUnitSphere();

    return Normalize(v);
}

inline Vec3 RandomOnHemisphere(const Vec3 &normal) {
    Vec3 v = RandomNormalizedVector();
    if (Dot(v, normal) > 0.0) {
        // on unit sphere (in the same hemisphere as the normal vector)
        return v;
    }

    return -v;
}

inline Vec3 RandomInUnitDisk() {
    for (;;) {
        Vec3 p(RandomF64(-1,1), RandomF64(-1,1), 0);
        if (p.LengthSq() < 1)
            return p;
    }
}

inline Vec3 Reflect(const Vec3 &v, const Vec3 &n) {
    return v - 2.0*Dot(v, n)*n;
}

inline Vec3 Refract(Vec3 uv, Vec3 n, f64 etai_over_etat) {
    f32 thetaCos = fmin(Dot(-uv, n), 1.0);
    Vec3 rPerp = etai_over_etat * (uv + thetaCos*n);
    Vec3 rParallel = -sqrt(fabs(1.0 - rPerp.LengthSq())) * n;

    return rPerp + rParallel;
}

using Color = Vec3;

class Ray {
public:
    Ray() = default;

    Ray(Vec3 origin, Vec3 dir): origin_{origin}, dir_{dir} {}

    Vec3 At(f64 t) const {
        return origin_ + t*dir_;
    }
    
    Vec3 origin_;
    Vec3 dir_;  
};


struct Interval {
    f64 min;
    f64 max;

    Interval(): min(+Infinity), max(-Infinity) {}

    Interval(f64 min, f64 max): min{min}, max{max} {}

    f64 Size() const { return max - min; }

    bool Contains(f64 x) const {
        return min <= x && x <= max;
    }

    bool Surrounds(f64 x) const {
        return min < x && x < max;
    }

    f64 Clamp(f64 x) const {
        if (x < min) return min;
        if (x > max) return max;

        return x;
    }

    static const Interval empty;
    static const Interval universe;
};

const Interval Interval::empty = Interval(+Infinity, -Infinity);
const Interval Interval::universe = Interval(-Infinity, +Infinity);

class Material;

struct HitRecord {
    Vec3 p;
    Vec3 normal;
    f64  t;
    std::shared_ptr<Material> material;
    bool frontFace;

    void SetFaceNormat(const Ray &r, Vec3 outwardNormal) {
        frontFace = Dot(r.dir_, outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Material {
public:
    virtual ~Material() = default;

    virtual bool Scatter(
            const Ray &r, const HitRecord &hit, Vec3 &attenuation, Ray &scattered) const {
        return false;
    }
};

class Lambertian : public Material {
public:
    Lambertian(Vec3 albedo): albedo_{albedo} {}

    bool Scatter(
            const Ray &r, const HitRecord &hit, Vec3 &attenuation, Ray &scattered) const override {
        
        Vec3 dir = hit.normal + RandomNormalizedVector();
       
        if (dir.NearZero())
            dir = hit.normal;

        scattered = Ray(hit.p, dir);
        attenuation = albedo_;

        return true;
    }
private:
    Vec3 albedo_;
};

class Metal : public Material {
public:
    Metal(Vec3 albedo, f64 fuzz): albedo_{albedo}, fuzz_(fuzz < 1 ? fuzz : 1) {}

    bool Scatter(
            const Ray &r, const HitRecord &hit, Vec3 &attenuation, Ray &scattered) const override {

        Vec3 reflected = Reflect(r.dir_, hit.normal);
        
        reflected = Normalize(reflected) + (fuzz_ * RandomNormalizedVector());
        scattered = Ray(hit.p, reflected);
        attenuation = albedo_;

        return (Dot(scattered.dir_, hit.normal) > 0);
    }
private:
    Vec3 albedo_;
    f64 fuzz_;
};

class Dielectric : public Material {
public:
    Dielectric(f64 refractionIndex): refractionIndex_{refractionIndex} {}

    bool Scatter(
            const Ray &r, const HitRecord &hit, Vec3 &attenuation, Ray &scattered) const override {
        attenuation = Vec3(1.0, 1.0, 1.0);
        f64 ri = hit.frontFace ? (1.0/refractionIndex_) : refractionIndex_;

        Vec3 unitDir = Normalize(r.dir_);
        f64 thetaCos = fmin(Dot(-unitDir, hit.normal), 1.0);
        f64 thetaSin = sqrt(1.0 - thetaCos*thetaCos);

        bool cannotRefract = ri * thetaSin > 1.0;
        Vec3 dir;
        if (cannotRefract || Reflectance(thetaCos, ri) > RandomF64())
            dir = Reflect(unitDir, hit.normal);
        else
            dir = Refract(unitDir, hit.normal, ri);


        scattered = Ray(hit.p, dir);

        return true;
    }
private:
    static f64 Reflectance(f64 cosine, f64 refractionIndex) {
        auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
        r0 = r0*r0;

        return r0 + (1-r0)*pow((1 - cosine), 5);
    }

    f64 refractionIndex_;
};

class Hittable {
public:
    virtual ~Hittable() = default;

    virtual bool Hit(const Ray &r, Interval ray_t, HitRecord &hit) const = 0;
};

class Sphere : public Hittable {
public:
    Sphere(Vec3 center, f64 radius, std::shared_ptr<Material> mat): center_{center}, radius_{fmax(0, radius)}, material_{mat} {}
    
    bool Hit(const Ray &r, Interval ray_t, HitRecord &hit) const override {
        Vec3 oc = center_ - r.origin_;
        f64 a = r.dir_.LengthSq();
        f64 h = Dot(r.dir_, oc);
        f64 c = oc.LengthSq() - radius_*radius_;

        f64 discriminant = h*h - a*c;

        if (discriminant < 0) {
            return false;
        } 

        f64 sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the range [tmin, tmax]
        f64 root = (h - sqrtd) / a;
        if (!ray_t.Surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.Surrounds(root))
                return false;
        }

        hit.p = r.At(root);
        Vec3 outwardNormal = (hit.p - center_) / radius_;
        hit.t = root;
        
        hit.SetFaceNormat(r, outwardNormal);
        hit.material = material_;
        return true;
    }
    
private:
    Vec3 center_;
    f64 radius_;
    std::shared_ptr<Material> material_;
};


class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;

    HittableList() = default;
    HittableList(const std::shared_ptr<Hittable> &object) { Add(object); }

    void Add(std::shared_ptr<Hittable> object) {
        objects.push_back(object);
    }

    bool Hit(const Ray &r, Interval ray_t, HitRecord &hit) const override {
        HitRecord temp;
        bool result = false;
        f64 closest = ray_t.max;

        for (const auto &object : objects) {
            if (object->Hit(r, Interval(ray_t.min, closest), temp)) {
                result = true;
                closest = temp.t;
                hit = temp;
            }
        }

        return result;
    }
};

Vec3 SampleSquare() {
    // Vector to a random point in the [-0.5, -0.5] - [+0.5, +0.5]
    return Vec3(RandomF64() - 0.5, RandomF64() - 0.5, 0.0);
}

inline f64 LinearToGamma(f64 c) {
    if (c > 0)
        return sqrt(c);

    return 0;
}

void WriteColor(std::ostream &out, const Color pixel) {
    auto r = LinearToGamma(pixel.x);
    auto g = LinearToGamma(pixel.y);
    auto b = LinearToGamma(pixel.z);

    static const Interval intensity(0.000, 0.999);
    
    int ir = int(256 * intensity.Clamp(r));
    int ig = int(256 * intensity.Clamp(g));
    int ib = int(256 * intensity.Clamp(b));

    std::cout << ir << ' ' << ig << ' ' << ib << '\n';
}

class Camera {
public:
    void Render(const Hittable &world) {
        Init();

        std::cout << "P3\n" << imageW_ << ' ' << imageH_ << "\n255\n";

        for (int j = 0; j < imageH_; j++) {
            for (int i = 0; i < imageW_; i++) {
                
                Vec3 pixelColor(0,0,0);
                for (int sample = 0; sample < samplesPerPixel_; sample++) {
                    Ray r = GetRay(i, j);
                    pixelColor += GetRayColor(r, maxDepth_, world);
                }
                WriteColor(std::cout, pixelSamplesScale_ * pixelColor);
            }
        }
    }

private:
    void Init() {
        imageH_ = int(imageW_ / ratio_);
        imageH_ = (imageH_ < 1) ? 1 : imageH_;
        
        pixelSamplesScale_ = 1.0 / samplesPerPixel_;

        center_ = lookfrom_;

        f64 theta = DegToRad(vfov_);
        f64 h = tan(theta/2);

        f64 viewportH = 2 * h * focusDist;
        f64 viewportW = viewportH * (f64(imageW_)/imageH_);

        w_ = Normalize(lookfrom_ - lookat_);
        u_ = Normalize(Cross(vup_, w_));
        v_ = Cross(w_, u_);

        Vec3 viewportU = viewportW * u_;
        Vec3 viewportV = viewportH * -v_;

        pixelDeltaU_ = viewportU / imageW_;
        pixelDeltaV_ = viewportV / imageH_;
        
        Vec3 viewportUpperLeft = center_ - (focusDist * w_) - viewportU/2 - viewportV/2;

        pixelLoc_ = viewportUpperLeft + 0.5 * (pixelDeltaU_ + pixelDeltaV_);

        auto defocusRadius = focusDist * tan(DegToRad(defocusAngle / 2));
        defocusDiskU = u_ * defocusRadius;
        defocusDiskV = v_ * defocusRadius;
    }

    Ray GetRay(int i, int j) const {
        Vec3 offset = SampleSquare();
        auto pixelSample = pixelLoc_ +
            ((i + offset.x)) * pixelDeltaU_ +
            ((j + offset.y)) * pixelDeltaV_;

            Vec3 rayOrigin = (defocusAngle <= 0) ? center_ : DefocusDiskSample();
            Vec3 rayDirection = pixelSample - rayOrigin;

            return Ray(rayOrigin, rayDirection);

    }

    Vec3 DefocusDiskSample() const {
        auto p = RandomInUnitDisk();
        return center_ + (p.x * defocusDiskU) + (p.y * defocusDiskV);
    }

    Color GetRayColor(const Ray &r, int depth, const Hittable &world) {
        if (depth <= 0)
            return Vec3(0, 0, 0);

        HitRecord hit;

        if (world.Hit(r, Interval(0.001, Infinity), hit)) {
            Ray scattered;
            Vec3 attenuation;
            if (hit.material->Scatter(r, hit, attenuation, scattered))
                return attenuation * GetRayColor(scattered, depth-1, world);
            return Vec3(0,0,0);
        }

        Vec3 unitDir = Normalize(r.dir_);
        auto a = 0.5*(unitDir.y + 1.0);
        
        return (1.0-a)*Color(1.0, 1.0, 1.0) + a*Color(0.5, 0.7, 1.0);
    }
public:
    f64 ratio_ = 1.0;
    int imageW_ = 100;
    int samplesPerPixel_ = 10;
    int maxDepth_ = 10;
    f64 vfov_ = 90;
    Vec3 lookfrom_ = Vec3(0,0,0);
    Vec3 lookat_ = Vec3(0,0,-1);
    Vec3 vup_ = Vec3(0,1,0);

    f64 defocusAngle = 0; // variation angle of rays through each pixel
    f64 focusDist = 10; // distance from camera lookfrom to plane of focus
private:
    int imageH_;
    f64  pixelSamplesScale_;
    Vec3 center_;
    Vec3 pixelLoc_;
    Vec3 pixelDeltaU_;
    Vec3 pixelDeltaV_;
    Vec3 u_, v_, w_; // camera frame basis
    Vec3 defocusDiskU;
    Vec3 defocusDiskV;

};

int main() {
    HittableList world;
  
    auto materialGround = std::make_shared<Lambertian>(Vec3(0.5, 0.5, 0.5));
    world.Add(std::make_shared<Sphere>(Vec3(0, -1000, 0), 1000, materialGround));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto randMat = RandomF64();
            Vec3 center(a + 0.9*RandomF64(), 0.2, b + 0.9*RandomF64());

            if ((center - Vec3(4, 0.2, 0)).Length() > 0.9) {
                std::shared_ptr<Material> materialSphere;

                if (randMat < 0.8) {
                    
                    auto albedo = Vec3::Random() * Vec3::Random();
                    materialSphere = std::make_shared<Lambertian>(albedo);
                    world.Add(std::make_shared<Sphere>(center, 0.2, materialSphere));
                } else if (randMat < 0.95) {
                    auto albedo = Vec3::Random(0.5, 1);
                    auto fuzz = RandomF64(0, 0.5);
                    materialSphere = std::make_shared<Metal>(albedo, fuzz);
                    world.Add(std::make_shared<Sphere>(center, 0.2, materialSphere));
                } else {
                    materialSphere = std::make_shared<Dielectric>(1.5);
                    world.Add(std::make_shared<Sphere>(center, 0.2, materialSphere));
                }
            }
        }
    }

    auto mat1 = std::make_shared<Dielectric>(1.5);
    world.Add(std::make_shared<Sphere>(Vec3(0, 1, 0), 1.0, mat1));

    auto mat2 = std::make_shared<Lambertian>(Vec3(0.4, 0.2, 0.1));
    world.Add(std::make_shared<Sphere>(Vec3(-4, 1, 0), 1.0, mat2));

    auto mat3 = std::make_shared<Metal>(Vec3(0.7, 0.6, 0.5), 0.0);
    world.Add(std::make_shared<Sphere>(Vec3(4, 1, 0), 1.0, mat3));
    
    Camera camera;
    camera.ratio_ = 16.0 / 9.0;
    camera.imageW_ = 1200;
    camera.samplesPerPixel_ = 500;
    camera.maxDepth_ = 50;
    camera.vfov_ = 20;
    camera.lookfrom_ = Vec3(13, 2, 3);
    camera.lookat_ = Vec3(0, 0, 0);
    camera.vup_ = Vec3(0, 1, 0);

    camera.defocusAngle = 0.6;
    camera.focusDist = 10.0;

    camera.Render(world);

    return 0;  
}

