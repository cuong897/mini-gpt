import unicodedata

from minigpt.cleaning import clean_prose, remove_nested


PROSE = ('Người dân trong làng cùng chăm sóc khu vườn và trồng thêm những cây xanh bên đường. '
         'Mỗi buổi sáng, trẻ em đến trường học tập, còn người lớn làm việc trên cánh đồng gần nhà. '
         'Cuộc sống nơi đây gắn liền với dòng sông và những mùa thu hoạch trong năm.')


def test_nested_templates_do_not_remove_surrounding_prose():
    text = '{{Taxobox|x={{nested|v=3}}}}\n\n'+PROSE
    result, _ = clean_prose(text)
    assert result == PROSE
    assert remove_nested('trước {{a|{{b}}}} sau', '{{','}}') == 'trước   sau'


def test_reference_body_removed_but_visible_link_text_preserved():
    text = PROSE.replace('dòng sông', '[[Sông|dòng sông]]')+'<ref name="x">CITATION NOISE</ref>'
    result, _ = clean_prose(text)
    assert result == PROSE and 'CITATION' not in result


def test_short_heading_reference_tail_and_lists_are_excluded():
    text = 'Lịch sử\n\n'+PROSE+'\n\n* Một danh sách ngắn\n\nTham khảo\n\n'+PROSE.replace('Người dân','Các nhà văn')
    result, stats = clean_prose(text)
    assert result == PROSE and stats['reference_tail_removed'] == 1


def test_unclosed_markup_is_not_silently_accepted():
    result, _ = clean_prose(PROSE+' {{malformed')
    assert result == ''


def test_preserve_accents_dates_and_no_factual_rewrite():
    text = PROSE+' Trận đấu diễn ra vào năm 1995, trong bài viết về năm 1999.'
    result, _ = clean_prose(unicodedata.normalize('NFD', text))
    assert result == text


def test_duplicate_paragraphs_removed_without_dropping_unique_prose():
    other = PROSE.replace('Người dân trong làng', 'Các học sinh trong trường')
    result, stats = clean_prose(PROSE+'\n\n'+PROSE+'\n\n'+other)
    assert result == PROSE+'\n\n'+other
    assert stats['duplicate_paragraphs_removed'] == 1


def test_markup_tables_and_non_vietnamese_fragments_excluded():
    text = '{| class="wikitable"\n| noise\n|}\n\n'+PROSE+'\n\n'+('This is an English sentence with no Vietnamese accents. '*8)
    result, _ = clean_prose(text)
    assert result == PROSE
