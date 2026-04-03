#!/bin/bash

set -e

echo "=================================================="
echo " 🚀 OCP Onboarder - Full Build Pipeline"
echo "=================================================="

PKG_NAME="ocp_gcp_onboarder"
MODULE_NAME="core"
#VERSION="1.0.0"
VERSION=$(date +%Y.%m.%d.%H%M)

echo ""
echo "📦 Step 1: Validate project structure..."

if [ ! -d "$PKG_NAME" ]; then
  echo "❌ Folder '$PKG_NAME' not found"
  exit 1
fi

if [ ! -f "$PKG_NAME/$MODULE_NAME.py" ]; then
  echo "❌ File '$PKG_NAME/$MODULE_NAME.py' not found"
  exit 1
fi

echo "   ✅ Structure OK"

echo ""
echo "🧹 Step 2: Clean old builds..."
rm -rf build dist *.egg-info
echo "   ✅ Cleaned"

echo ""
echo "📦 Step 3: Install local build deps..."
pip install --upgrade pip setuptools wheel cython >/dev/null
echo "   ✅ Dependencies installed"

echo ""
echo "🔐 Step 4: Generate setup.py..."

cat > setup.py <<EOF
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension("$PKG_NAME.$MODULE_NAME", ["$PKG_NAME/$MODULE_NAME.py"])
]

setup(
    name="ocp-cloud-intake-agent",
    version="$VERSION",
    description="Jira-based GCP onboarding automation agent",
    author="Praveen HA",
    author_email="praveen.balla41@gmail.com",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    package_data={"$PKG_NAME": ["*.so"]},
    include_package_data=True,
    install_requires=["requests", "python-dotenv"],
    entry_points={
        "console_scripts": [
            "ocp-onboarder=$PKG_NAME.$MODULE_NAME:main"
        ]
    },
    zip_safe=False,
)
EOF

echo "   ✅ setup.py ready"

echo ""
echo "🐳 Step 5: Build inside manylinux Docker..."

docker run --rm -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 /bin/bash -c "
cd /io
/opt/python/cp312-cp312/bin/pip install setuptools wheel cython
/opt/python/cp312-cp312/bin/python setup.py bdist_wheel
"

echo "   ✅ Wheel built (raw)"

echo ""
echo "🔧 Step 6: Repair wheel (auditwheel)..."

docker run --rm -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 /bin/bash -c "
/opt/python/cp312-cp312/bin/pip install auditwheel
auditwheel repair /io/dist/*.whl -w /io/dist/
"

echo "   ✅ manylinux wheel ready"

echo ""
echo "📦 Step 7: Final output..."

ls dist/*manylinux*.whl

echo ""
echo "=================================================="
echo " ✅ BUILD COMPLETE"
echo "=================================================="

echo ""
echo "🚀 Next step: Upload to PyPI"
echo "twine upload dist/*manylinux*.whl"
echo ""
