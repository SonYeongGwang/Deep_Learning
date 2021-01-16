from imutils import paths

input_path = '/home/a/animals/test_set'
imagePaths = list(paths.list_images(input_path))
print(imagePaths, len(imagePaths))

# list_images function can find all images in the sub directory