# CoQA-Challenge
NLP 585 Course Project

# How to run

## Evaluation:
BiDAF++ model:
-  Dev set: ./bidafplus_dev_evaluate.sh
- Test set: ./bidafplus_test_evaluate.sh

Seq2Seq model:
-  Dev set: ./seq2seq_dev_evaluate.sh
- Test set: ./seq2seq_test_evaluate.sh

## Generating Predictions:
BiDAF++ model:
-  Dev set: ./bidafplus_dev_predict.sh
- Test set: ./bidafplus_test_predict.sh

Seq2Seq model:
-  Dev set: ./seq2seq_dev_predict.sh
- Test set: ./seq2seq_test_predict.sh

Top 50 worst F1 matches: (story_id, turn_id, predicted_ans, gold_ans)

- 1 (('3a4tn5196kisae3e88uoqj60g17chl', 10, 'october', 'Microsoft Word is a word processor developed by Microsoft. It was first released on October 25, 1983 under the name "Multi-Tool Word" for Xenix systems. Subsequent versions were later written for several other platforms including IBM PCs running DOS (1983), Apple Macintosh running Classic Mac OS (1985), AT&T Unix PC (1985), Atari ST (1988), OS/2 (1989), Microsoft Windows (1989), SCO Unix (1994), and macOS (2001). Commercial versions of Word are licensed as a standalone product or as a component of Microsoft Office, Windows RT or the discontinued Microsoft Works suite. Microsoft Word Viewer and Office Online are freeware editions of Word with limited features.'), 0.02)
- 2 (('3ffj6vril1o8chji2ajpvu5e7ko0iy', 6, 'he would make himself cry in less than a second .', 'He went over to Pipo, took off his hat, put it face-up on the ground, and started crying'), 0.07692307692307691)
- 3 (('3ydtzai2wxgebz5ld4llfye57wl41m', 8, 'instruct sikes to apply the brakes and the emergency brake at the same time', 'dispatchers talked him through instructions on how he might be able to stop the car'), 0.08)
- 4 (('3pb5a5bd0v68y1d7xl4vpx2l0o57gm', 5, 'full of praise', "Webb's tally of 14 yellow cards -- including the red shown to Netherlands defender Johnny Heitinga in extra-time - was a record for the World Cup final."), 0.08)
- 5 (('3z4xg4zf48rnk1dgw0w5rjybe0b8xa', 8, 'cell phone networks , wireless sensor networks , and terrestrial microwave networks .', "it networks internet devices within a relatively small area, that is generally within a person's reach."), 0.08333333333333333)
- 6 (('3bdcf01ogxu7zdn9vlrbf2rqzwplyf', 4, 'because of the way he struck his iron shoes on the stones .', 'to take charge of her cattle and horse and ranges, and save them'), 0.08695652173913043)
- 7 (('3fprzhyepy79ff2fk40rchtfi333v2', 13, 'colorado of jesus christ of latter-day saints', 'Colorado to the east, Wyoming to the northeast, Idaho to the north, Arizona to the south, and Nevada to the west'), 0.08695652173913043)
- 8 (('37qw5d2zrgmfokrh2qqisbhjy9hs85', 6, 'to help meet the needs of local community members', 'a student who got frostbite riding his bike to school because he didnt have gloves'), 0.09090909090909091)
- 9 (('3copxfw7xbc26tdqjyjrnblz6w4kpe', 13, 'because of that feeling , he picked up the phone and all of a sudden he passed out', 'he was in such a nice place'), 0.09523809523809522)
- 10 (('3u84xhcdicdb6vqtlfud7syhk5jz4f', 12, 'castles', 'many castles, palaces, town halls, guild halls, universities and to a less prominent extent, private dwellings, such as dorms and rooms'), 0.09523809523809523)
- 11 (('3wi0p0ii61sf40nv491totqoocvrdf', 7, "i do n't think i can rest until i have seen the lode .", 'She needs to rest for two or three days.'), 0.09523809523809525)
- 12 (('3eg49x351uc0gnus3lz7752k7xo6x8', 9, "when mary 's father saw mary 's sewing things and gave them to the poor children", 'After they saw her crying.'), 0.1)
- 13 (('3ve8ayvf8mx6kfmvw6qjlcy4azrf8a', 14, "he 's all true", "he didn't like to tell things to people who wouldn't believe what he told them"), 0.10526315789473685)
- 14 (('3dr23u6we5exclen4th8uq9rc7jetu', 1, 'messianism , heaven and hell ,', 'It combines a cosmogonic dualism and eschatological monotheism in a manner unique [...] among the major religions of the world"'), 0.10526315789473685)
- 15 (('32scwg5hih4v7es1hupqdsgh52d6pq', 6, 'because of this the summer in west texas', 'Because the temperature is often over 100, which causes them to break.'), 0.1111111111111111)
- 16 (('3io1lgzlk9xa1mtkvdnfr6lrgkd68q', 2, 'he was too different .', 'Because he spent many years in a special hospital for people with mental health problems.'), 0.11111111111111112)
- 17 (('3nkqq8o39y57ksfc83wyt4d8v9wudf', 9, "he 's a minor and it 's not clear if he 'll be charged as an adult .", "he's a minor"), 0.11764705882352941)
- 18 (('3i33ic7zwf20293y59vqxkaaq2ka2o', 14, 'yes', "yes go to David's own website. Now David sells hundreds of guitar lesson videos each week"), 0.11764705882352941)
- 19 (('30mvjzjnhmdm3mr1koni06l7mw8j9r', 13, "nick thought he could n't find his blue socks to wear to school .", 'He did it himself'), 0.11764705882352941)
- 20 (('3f6kkywmnb1up2v3b2kcf9lem27ndx', 5, 'taking time to tease one another about their receding hairlines .', 'They were friends since their early years'), 0.11764705882352941)
- 21 (('3xxu1swe8mvt6z0kqmrcewhvui4a01', 11, "she said that `` olaf must surely have taken a longer walk than usual that day . ''", 'she became uneasy'), 0.11764705882352941)
- 22 (('3wq3b2kge8gywyqusjv8nckbhq11bs', 14, 'the films are the longest continually running film series to date', 'television, radio, comic strip, video games and film'), 0.11764705882352941)
- 23 (('3cfvk00fwll5gtd3p2wjwb7x1wx6lm', 6, "`` hurrah , but it 's hot work ! ''", '"Nonsense! Now, both together, and put all your muscle into it,'), 0.11764705882352942)
- 24 (('33iztu6j81153lspay2a8aycqqaxsj', 12, 'when he sat down to an 18-course banquet', 'Because they were there to discuss world food shortages.'), 0.125)
- 25 (('3vfjci1k4zzigkxm6z21uetl0zlgrd', 12, 'the man yelled at me for the door', 'The man wanted make sure they didnt get out or hurt'), 0.125)
- 26 (('3of2m9aatgowkxfw67hte9ndgvdkz1', 9, 'started making spooky sounds and scared henry', 'Make Henry think one side of the lake was scarey'), 0.125)
- 27 (('3wi0p0ii61sf40nv491totqoocvrdf', 8, 'you will have to try .', "I don't think I can rest until I have seen the lode."), 0.12500000000000003)
- 28 (('3kjyx6qcm9bk0t44npsesoa4exnvjy', 11, 'he was accused by police', 'He thinks the authorities are trying to threaten him through her'), 0.13333333333333333)
- 29 (('3kxir214i4gl0knhw8lzkhoazwm42e', 2, 'the city covers a land area of 142.8 square miles', 'combination of three cities and their universities'), 0.13333333333333333)
- 30 (('3cfvk00fwll5gtd3p2wjwb7x1wx6lm', 11, 'he was always a pretty fair oarsman', '"He must be almost exhausted to row like that."'), 0.13333333333333333)
- 31 (('3gm6g9zbknxvo960lr5r7ye0lcttmn', 14, 'that shift is reflected in his comedy', 'That he was "seven years late for work."'), 0.13333333333333333)
- 32 (('33m4ia01qg1t26scv925i0tg3otrxq', 7, "julie did n't want all dolls could n't .", "she didn't want her friends to know"), 0.13333333333333333)
- 33 (('3gna64guze4komt2coualrsrfyqq5e', 9, 'as long ago as 650 thousand years ago', 'The body louse specifically lives in clothing, and diverge from head lice about 107,000 years ago, suggesting that clothing existed at that time.'), 0.13333333333333333)
- 34 (('35bldd71i6xa08985bv0giyuxzbvz2', 15, 'it was felt that he must be', 'because some impression might be made upon her'), 0.13333333333333333)
- 35 (('34majl3qp4nal2j008z43rt25ds431', 3, 'elizabeth von arnim', 'Elizabeth, her husband, her three tiny daughters, various servants and some visitors'), 0.13333333333333333)
- 36 (('3jv9lgbjwtefj756e7lx0jogqj9gor', 11, 'because he was great that he was quite handsome', 'Because of his good-looking face'), 0.14285714285714285)
- 37 (('3ryc5t2d73totxql9isoon7d2x4rp3', 7, 'he was back on dave', 'He could tell him something he wants to hear'), 0.14285714285714285)
- 38 (('3rsdurm96amtt7dhez472716q5xey6', 11, 'afraid to tell her father', 'he could tell when she looked him in the eye'), 0.14285714285714285)
- 39 (('3igi0vl647kltzms1bysq3xdrrnonr', 8, 'it became a constituent part of the bohemian crown', 'it was incorporated into the early Polish state'), 0.14285714285714285)
- 40 (('3io1lgzlk9xa1mtkvdnfr6lrgkd68q', 9, 'he finished medical school', 'He often dressed up like a clown to make the children laugh.'), 0.14285714285714288)
- 41 (('3strjbfxowr0yl6x0fsbslmwv1ukts', 5, 'to play games and play outside', 'He wanted her to be with her friends'), 0.14285714285714288)
- 42 (('3i3wadaz9q4h3agmxb26wmxr009o5b', 14, 'noticed that there was something attached to one of its claws .', 'went to police'), 0.14285714285714288)
- 43 (('3bxqmrhwkzyaomlplwv1cu024nzum8', 6, 'the whole of your effects is the whole of your conception of the object .', 'The nature of knowledge.'), 0.14285714285714288)
- 44 (('3wr9xg3t63bsmlkn2k2ug85iaoa47k', 8, 'fifteen', 'Approximately 2,600 athletes from 82 nations participated in 86 events in fifteen disciplines'), 0.14285714285714288)
- 45 (('3gu1kf0o4i11dq9wdl6yo829jzppbn', 1, 'because he spent so much time thinking of odd stories', "He had writer's block"), 0.14285714285714288)
- 46 (('39zsfo5ca8wknef4izi9w28lzpauj1', 14, 'one day looks like with a splinter', 'so he could remember what the foot looked like'), 0.14285714285714288)
- 47 (('3ls2amnw5fq6wwzkh3q9uxsivykqoh', 2, 'he fed one of them seaweed', "Because he didn't have to go to school."), 0.14285714285714288)
- 48 (('3zr9aiqjub9e4ak3hlhl1tvv27p40z', 12, 'rail its position as a regional centre of power', 'the Finger Plan fostered the development of housing and businesses along the five urban railway routes stretching out from the city centre.'), 0.15384615384615383)
- 49 (('3rjsc4xj10uw0to3vq0v6l191rp05g', 10, 'in the late 1800s', 'In France cooks changed the ice recipe and made ice cream'), 0.15384615384615383)
- 50 (('3kb8r4zv1e7v0dgxa2gbuzohi37gb5', 11, 'i am lisa , saying that he was not a good father .', 'he embraced her'), 0.15384615384615383)
