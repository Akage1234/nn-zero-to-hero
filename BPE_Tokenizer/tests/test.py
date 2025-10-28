from ..GPT4Tokenizer import GPT4Tokenizer
import tiktoken

test_strings = [
    "hello world!!!? (안녕하세요!) lol123 😉",
    "abc@gmail.com",
    '''Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. Her versatile artistry, songwriting, and entrepreneurship have influenced the music industry, popular culture, and politics, and her life is a subject of widespread media coverage.
    Swift began professional songwriting at 14 and signed with Big Machine Records in 2005 to become a country singer. She released six studio albums under the label, four of them to country radio, starting with Taylor Swift (2006). Her next, Fearless (2008), explored country pop, and its singles "Love Story" and "You Belong with Me" catapulted her to mainstream fame. Speak Now (2010) infused rock influences, while Red (2012) experimented with electronic elements and featured Swift's first Billboard Hot 100 number-one song, "We Are Never Ever Getting Back Together". She departed from her country image with 1989 (2014), a synth-pop album supported by the chart-topping songs "Shake It Off", "Blank Space", and "Bad Blood". Media scrutiny inspired the hip-hop-influenced Reputation (2017) and its number-one single "Look What You Made Me Do".
    After signing with Republic Records in 2018, Swift released the eclectic pop album Lover (2019) and the autobiographical documentary Miss Americana (2020). She explored indie folk styles on the 2020 albums Folklore and Evermore, subdued electropop on Midnights (2022), and re-recorded four albums subtitled Taylor's Version after a dispute with Big Machine. These albums spawned the number-one songs "Cruel Summer", "Cardigan", "Willow", "Anti-Hero", "All Too Well", and "Is It Over Now?". Her Eras Tour (2023–2024) and its accompanying concert film became the highest-grossing tour and concert film of all time, respectively. Swift has directed several music videos and films such as Folklore: The Long Pond Studio Sessions (2020) and All Too Well: The Short Film (2021).
    One of the world's best-selling musicians, with over 200 million records sold as of 2019, Swift has been named Global Recording Artist of the Year three times by the International Federation of the Phonographic Industry, whereas six of her albums have opened with over a million sales in a week. She is the highest-grossing female touring act, the most-streamed woman on Spotify and Apple Music, and the first billionaire with music as the main source of income. The 2023 Time Person of the Year, Swift has appeared on lists such as Rolling Stone's 100 Greatest Songwriters of All Time, Billboard's Greatest of All Time Artists, and Forbes' World's 100 Most Powerful Women. Her accolades include 14 Grammy Awards (featuring a record four Album of the Year wins), a Primetime Emmy Award, 40 American Music Awards, 40 Billboard Music Awards, and 23 MTV Video Music Awards.
    Life and career
    Early life
    ''',
    "<|endoftext|>hello world"
]

tiktoken_enc = tiktoken.get_encoding('cl100k_base')

gpt_tokenizer = GPT4Tokenizer()
for i, text in enumerate(test_strings):
    tiktoken_ids = tiktoken_enc.encode(text, allowed_special='all')
    tiktoken_text = tiktoken_enc.decode(tiktoken_ids)

    gpt_ids = gpt_tokenizer.encode(text, allowed_special='all')
    gpt_output = gpt_tokenizer.decode(gpt_ids)
    print(f"Test {i}:", 'passed' if tiktoken_text == gpt_output and tiktoken_ids == gpt_ids else 'fail')



