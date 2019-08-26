rm -rf bayestuner.egg-info
rm -rf build
rm -rf dist

python3 setup.py sdist bdist_wheel

python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

sleep 50

pip uninstall bayestuner

pip install -i https://test.pypi.org/simple/ bayestuner
