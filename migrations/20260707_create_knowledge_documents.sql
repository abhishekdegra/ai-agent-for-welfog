-- Step 1 migration: create knowledge_documents in MySQL
-- Safe to run multiple times.

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(512) NOT NULL,
    category VARCHAR(128) NOT NULL,
    content LONGTEXT NOT NULL,
    language VARCHAR(32) NOT NULL DEFAULT 'en',
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    version INT NOT NULL DEFAULT 1,
    index_status VARCHAR(32) NOT NULL DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_kd_category_status_lang (category, status, language),
    INDEX idx_kd_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;
