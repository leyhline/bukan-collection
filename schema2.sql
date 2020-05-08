CREATE TABLE title (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT,
    kanji VARCHAR(50) NOT NULL,
    hiragana VARCHAR(50) NOT NULL,
    romanji VARCHAR(50) NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (kanji)
);

CREATE TABLE book (
    id INT UNSIGNED NOT NULL,
    released DATE NOT NULL,
    title_id INT UNSIGNED NOT NULL,
    original_id VARCHAR(20) NOT NULL,
    era_name VARCHAR(2) NOT NULL,
    era_year TINYINT UNSIGNED,
    year SMALLINT UNSIGNED,
    estimate BOOLEAN,
    nr_books INT UNSIGNED,
    pages_per_scan TINYINT UNSIGNED NOT NULL,
    aspect BINARY(1) NOT NULL,
    nr_scans INT UNSIGNED NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (title_id) REFERENCES title (id),
    INDEX (year)
);

CREATE TABLE page (
    id INT UNSIGNED NOT NULL,
    book_id INT UNSIGNED NOT NULL,
    page INT UNSIGNED NOT NULL,
    lr BINARY(1) NOT NULL,
    filename CHAR(21),
    PRIMARY KEY (id),
    FOREIGN KEY (book_id) REFERENCES book (id)
);

CREATE TABLE feature (
    id BIGINT UNSIGNED NOT NULL,
    page_id INT UNSIGNED NOT NULL,
    feature_nr INT UNSIGNED NOT NULL,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    size FLOAT NOT NULL,
    angle FLOAT NOT NULL,
    response FLOAT NOT NULL,
    octave TINYINT UNSIGNED NOT NULL,
    class_id TINYINT UNSIGNED NOT NULL,
    descriptor BINARY(61) NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (page_id) REFERENCES page (id),
    UNIQUE (page_id, feature_nr)
);

CREATE TABLE dmatch (
    id BIGINT UNSIGNED NOT NULL,
    src_feature BIGINT UNSIGNED NOT NULL,
    dst_feature BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (src_feature) REFERENCES feature (id),
    FOREIGN KEY (dst_feature) REFERENCES feature (id),
    UNIQUE (src_feature, dst_feature)
);

CREATE TABLE pagepair (
    id BIGINT UNSIGNED NOT NULL,
    first_page INT UNSIGNED NOT NULL,
    second_page INT UNSIGNED NOT NULL,
    nr_matches INT UNSIGNED NOT NULL,
    h11 FLOAT,
    h12 FLOAT,
    h13 FLOAT,
    h21 FLOAT,
    h22 FLOAT,
    h23 FLOAT,
    h31 FLOAT,
    h32 FLOAT,
    h33 FLOAT,
    PRIMARY KEY (id),
    FOREIGN KEY (first_page) REFERENCES page (id),
    FOREIGN KEY (second_page) REFERENCES page (id),
    UNIQUE (first_page, second_page)
);