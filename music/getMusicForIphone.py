import os
import shutil
import eyed3

source = "Z:/arxiv"
destination = "Z:/exportable"

if not os.path.exists(destination):
    os.makedirs(destination)

total = 0
ok = 0

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(source):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)
        shutil.copy(root + os.sep + file, str(destination))
        total += 1
        
        try:
            audio_file = eyed3.load(str(destination) + os.sep + file)
            audio_file.initTag()
            audio_file.tag.artist = str(os.path.basename(root))
            audio_file.tag.album = str(os.path.basename(root))
            audio_file.tag.album_artist = str(os.path.basename(root))
            audio_file.tag.title = ""
            audio_file.tag.track_num = total
            audio_file.tag.save()
            ok += 1
        except (AttributeError, NotImplementedError):
            print('*')
        except BaseException as err:
            print('*')
            print(f"Unexpected {err=}, {type(err)=}")

print("\n\nNot written " + str(total - ok) + " files")
print("Written " + str(ok) + " files")
print("Total " + str(total) + " files")         