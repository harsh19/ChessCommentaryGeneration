1. getGameLinks.py was originally used to get the set of all current game links from gameknot.com. This is saved as saved_files/saved_links.p. Note however that since newer games keep getting added 
   to gameknot.com, this is not exactly reproducible. Hence, we provide saved_files/saved_links.p directly for reproducibility. 
   
   saved_files/saved_links.p is a list of 11578 URLs, each one corresponding to the first page of a distinct game.

2. Some games finish in one page. But some others are multi-page. extra_pages.p stores the count of pages which make up the game, for each of the games linked to by game links.
   saved.
   extra_pages.p is a list of 11578 integers.
   If extra_pages[i]=0, that means that game 0 has 0 extra pages [1 game page in total]. If its 2, that means it has 2 extra pages [3 games pages in total]

3. trainTestSplit.py: Splits saved_files/saved_links.p into saved_files/train_links.p, saved_files/valid_links.p and saved_files/test_links.p. The splitting is done based on a fixed seed. However,      since fixed seeds may sometimes have different effects based on python versions etc, we also provide these files beforehand in the repo - so you don't really need to run trainTestSplit.py

4. Now, run: 
   ``` python run_all.py 0 11577 1  ```
   This downloads all pages from index 0-11577, using a single thread. If you want to speed this up, increase the third parameter from 1 
   to n to use n threads in parallel [Don't use a very large n, stick to a value <5] 

   This will crawl the webpages associated with each of the game links, and save them under saved_files/ . Note that pages which are multi-page [Page 1 or beyond, not Page 0] are saved with a 
   file name containing an additional suffix indicating their number.

   Underneath the hood, this runs "python save_rendered_webpage.py $NUM" where $NUM is the game number for each page.

5. Now run the following command:
   ```python main.py html_parser```
   The command finds all .html files in saved_files/ and extracts object representations from them, saving them as pickled .obj files, with
   the same prefix as the .html files, in the ./outputs/ directory.

6. Now run: 
```
   python preprocess.py train
   python preprocess.py valid
   python preprocess.py test
```

   This will create requisite train,valid and test files under saved_files/ with the following nomenclature:
   1. train.single.che contains one game situation per line. train.single.en contains the corresponding comments
   2. ".che" and ".en" correspond to game situation and english-comment respectively. Given the same prefix, these files have the same number of lines
   3. ".single" and ".multi" denote single-move situations vs multi-move situations. The second one occurs because comments are sometimes made on a sequence of moves rather than just one move.

   Format of .che file:
   The .che file contains Linearized Game State of Current Board <EOC> Linearized Game State of Previous Board <EOP> Move Specification <EOM> <EOMH> Raw Move Notation <EOM>
   1. Each Linearized Game State is a sequence of 64 tokens. The "eps" token indicates an empty square. The rest of the tokens are self-explanatory , of the form color_pieceName (e.g black_bishop)
   2. The Move Specification takes on slightly varied forms based on whether it was a simple move, or a move with a capture or check. Some of the forms mean the following:
            a. black black knight d5 e7 indicates that Black moved the black knight from square d5 to square e7
            b. Check moves are suffixed with an additional "check" at the end
            c. Capture moves have an additional capture "pieceColor pieceType" at the end. e.g white white queen d3 h6 check capture black knight indicates that White moved the white queen from d3 to h6, putting a check and also capturi               ng the black knight.
