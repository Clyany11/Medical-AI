SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;
CREATE DATABASE IF NOT EXISTS `aidoctor`
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
USE `aidoctor`;
-- ----------------------------
-- Table structure for feedback_expert_queue
-- ----------------------------
DROP TABLE IF EXISTS `feedback_expert_queue`;
CREATE TABLE `feedback_expert_queue`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `task_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `session_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `user_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'anonymous',
  `status` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'pending' COMMENT 'pending/in_review/completed',
  `feedback_reason` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `answer` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `query` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `bad_generation` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `milvus_trace` json NULL,
  `expert_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `expert_answer` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `notes` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `reviewed_at` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `idx_task_id`(`task_id`) USING BTREE,
  INDEX `idx_session_id`(`session_id`) USING BTREE,
  INDEX `idx_status`(`status`) USING BTREE,
  INDEX `idx_created_at`(`created_at`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 62 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '专家审核队列表' ROW_FORMAT = Dynamic;
-- ----------------------------
-- Table structure for feedback_interaction
-- ----------------------------
DROP TABLE IF EXISTS `feedback_interaction`;
CREATE TABLE `feedback_interaction`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `session_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `user_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'anonymous',
  `query` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `generation` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `source` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '',
  `cost_seconds` float NULL DEFAULT 0,
  `tokens_used` json NULL,
  `milvus_trace` json NULL,
  `feedback` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `expert_task_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `expert_answer` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_session_id`(`session_id`) USING BTREE,
  INDEX `idx_user_id`(`user_id`) USING BTREE,
  INDEX `idx_created_at`(`created_at`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '交互记录表, 替代 interactions.jsonl 的查询功能' ROW_FORMAT = Dynamic;
-- ----------------------------
-- Table structure for feedback_thumbs
-- ----------------------------
DROP TABLE IF EXISTS `feedback_thumbs`;
CREATE TABLE `feedback_thumbs`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `session_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `user_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'anonymous',
  `feedback_type` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'thumbs_up 或 thumbs_down',
  `feedback_reason` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '点踩原因: harmful/false_info/not_helpful/other',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '用户输入的反馈意见文本',
  `answer` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '被点踩的 AI 回复内容',
  `query` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `bad_generation` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL,
  `milvus_trace` json NULL,
  `tokens_used` json NULL,
  `cost_seconds` float NULL DEFAULT 0,
  `expert_task_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `data_value` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL,
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_session_id`(`session_id`) USING BTREE,
  INDEX `idx_feedback_type`(`feedback_type`) USING BTREE,
  INDEX `idx_created_at`(`created_at`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 148 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '点赞/点踩记录表' ROW_FORMAT = Dynamic;
-- ----------------------------
-- Table structure for user_chat_message
-- ----------------------------
DROP TABLE IF EXISTS `user_chat_message`;
CREATE TABLE `user_chat_message`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `session_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '外键，关联 shop_chat_sessions.id',
  `sender` char(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '发送者（\'user\' / \'bot\'）',
  `message_text` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '消息内容',
  `message_timestamp` datetime(0) NULL DEFAULT NULL COMMENT '消息时间',
  `message_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '消息类型（如文件files，文本text、图片pic、按钮button等，可选）',
  `is_prompt` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '是否为用户问题',
  `token_count` varchar(5000) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '消耗token数',
  `model_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '聊天使用的模型，gpt4o，doubao，deepseek',
  `platform_source` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '发送的使用机型：mobile，pc。(android，ios，web，pc)',
  `create_time` datetime(0) NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `create_by` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '创建人创建项目',
  `update_time` datetime(0) NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP(0) COMMENT '更新时间',
  `update_by` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '更新人创建项目',
  `is_deleted` tinyint(1) NULL DEFAULT 0 COMMENT '是否删除1-是0-否',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2449 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '表单收集数据结果' ROW_FORMAT = Dynamic;
-- ----------------------------
-- Table structure for user_chat_session
-- ----------------------------
DROP TABLE IF EXISTS `user_chat_session`;
CREATE TABLE `user_chat_session`  (
  `uid` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '会话唯一标识（可用于前端标记）之后商议一下，可以使用这个表的主key也就是id代替',
  `id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL COMMENT '页面使用的唯一id',
  `user_id` bigint(20) NULL DEFAULT NULL COMMENT '用户id，外键，关联 users表id，（目前不能为空，也就是匿名状态）',
  `username` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `started_at` datetime(0) NULL DEFAULT CURRENT_TIMESTAMP COMMENT '会话开始时间',
  `ended_at` datetime(0) NULL DEFAULT NULL COMMENT '会话结束时间-这个应该得不到，暂时不写',
  `language` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '语言偏好',
  `topic` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '对话主题标签',
  `create_time` datetime(0) NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `create_by` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '创建人创建项目',
  `update_time` datetime(0) NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP(0) COMMENT '更新时间',
  `update_by` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '更新人创建项目',
  `is_deleted` tinyint(1) NULL DEFAULT 0 COMMENT '是否删除1-是0-否',
  PRIMARY KEY (`uid`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 348 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '表单收集数据结果' ROW_FORMAT = Dynamic;
-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users`  (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `username` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `password_hash` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `nickname` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP(0),
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `username`(`username`) USING BTREE,
  UNIQUE INDEX `nickname`(`nickname`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 51 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_bin ROW_FORMAT = Dynamic;
SET FOREIGN_KEY_CHECKS = 1;