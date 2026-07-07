-- Step 3 migration: create cleaned/chunked knowledge table (no vectors yet)
-- Safe to run multiple times.

CREATE TABLE IF NOT EXISTS knowledge_document_chunks (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chunk_id VARCHAR(128) NOT NULL,
    doc_id BIGINT NOT NULL,
    version INT NOT NULL,
    chunk_no INT NOT NULL,
    title VARCHAR(512) NOT NULL,
    category VARCHAR(128) NOT NULL,
    language VARCHAR(32) NOT NULL,
    content LONGTEXT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_kdc_chunk_id (chunk_id),
    UNIQUE KEY uq_kdc_doc_ver_chunk (doc_id, version, chunk_no),
    INDEX idx_kdc_doc_ver (doc_id, version),
    INDEX idx_kdc_category_lang (category, language)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;
