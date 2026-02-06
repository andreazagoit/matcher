CREATE TYPE "public"."gender" AS ENUM('man', 'woman', 'non_binary');--> statement-breakpoint
CREATE TYPE "public"."assessment_status" AS ENUM('in_progress', 'completed');--> statement-breakpoint
CREATE TYPE "public"."member_role" AS ENUM('owner', 'admin', 'member');--> statement-breakpoint
CREATE TYPE "public"."member_status" AS ENUM('pending', 'active', 'suspended');--> statement-breakpoint
CREATE TABLE "users" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"first_name" text NOT NULL,
	"last_name" text NOT NULL,
	"email" text NOT NULL,
	"birth_date" date NOT NULL,
	"gender" "gender",
	"email_verified" boolean DEFAULT false NOT NULL,
	"image" text,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE "assessments" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"assessment_name" text NOT NULL,
	"answers" jsonb NOT NULL,
	"status" "assessment_status" DEFAULT 'completed' NOT NULL,
	"completed_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "profiles" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"psychological_desc" text,
	"values_desc" text,
	"interests_desc" text,
	"behavioral_desc" text,
	"psychological_embedding" vector(1536),
	"values_embedding" vector(1536),
	"interests_embedding" vector(1536),
	"behavioral_embedding" vector(1536),
	"assessment_version" real DEFAULT 1,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "profiles_user_id_unique" UNIQUE("user_id")
);
--> statement-breakpoint
CREATE TABLE "spaces" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" text NOT NULL,
	"slug" text NOT NULL,
	"description" text,
	"image" text,
	"client_id" text NOT NULL,
	"secret_key" text NOT NULL,
	"secret_key_hash" text NOT NULL,
	"redirect_uris" text[],
	"visibility" text DEFAULT 'public' NOT NULL,
	"join_policy" text DEFAULT 'open' NOT NULL,
	"access_token_ttl" text DEFAULT '3600',
	"refresh_token_ttl" text DEFAULT '2592000',
	"is_active" boolean DEFAULT true NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "spaces_slug_unique" UNIQUE("slug"),
	CONSTRAINT "spaces_client_id_unique" UNIQUE("client_id")
);
--> statement-breakpoint
CREATE TABLE "members" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"space_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"role" "member_role" DEFAULT 'member' NOT NULL,
	"status" "member_status" DEFAULT 'active' NOT NULL,
	"joined_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "posts" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"space_id" uuid NOT NULL,
	"author_id" uuid NOT NULL,
	"content" text NOT NULL,
	"media_urls" text[],
	"likes_count" integer DEFAULT 0,
	"comments_count" integer DEFAULT 0,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "authorization_codes" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"code" text NOT NULL,
	"client_id" text NOT NULL,
	"user_id" uuid NOT NULL,
	"redirect_uri" text NOT NULL,
	"scope" text NOT NULL,
	"state" text,
	"code_challenge" text,
	"code_challenge_method" text,
	"expires_at" timestamp NOT NULL,
	"used_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "authorization_codes_code_unique" UNIQUE("code")
);
--> statement-breakpoint
CREATE TABLE "access_tokens" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"token_hash" text NOT NULL,
	"jti" text NOT NULL,
	"client_id" text NOT NULL,
	"user_id" uuid,
	"scope" text NOT NULL,
	"expires_at" timestamp NOT NULL,
	"revoked_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "access_tokens_token_hash_unique" UNIQUE("token_hash"),
	CONSTRAINT "access_tokens_jti_unique" UNIQUE("jti")
);
--> statement-breakpoint
CREATE TABLE "refresh_tokens" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"token_hash" text NOT NULL,
	"jti" text NOT NULL,
	"access_token_id" uuid,
	"client_id" text NOT NULL,
	"user_id" uuid,
	"scope" text NOT NULL,
	"expires_at" timestamp NOT NULL,
	"revoked_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "refresh_tokens_token_hash_unique" UNIQUE("token_hash"),
	CONSTRAINT "refresh_tokens_jti_unique" UNIQUE("jti")
);
--> statement-breakpoint
ALTER TABLE "assessments" ADD CONSTRAINT "assessments_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "profiles" ADD CONSTRAINT "profiles_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "members" ADD CONSTRAINT "members_space_id_spaces_id_fk" FOREIGN KEY ("space_id") REFERENCES "public"."spaces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "members" ADD CONSTRAINT "members_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "posts" ADD CONSTRAINT "posts_space_id_spaces_id_fk" FOREIGN KEY ("space_id") REFERENCES "public"."spaces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "posts" ADD CONSTRAINT "posts_author_id_users_id_fk" FOREIGN KEY ("author_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "authorization_codes" ADD CONSTRAINT "authorization_codes_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "access_tokens" ADD CONSTRAINT "access_tokens_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "refresh_tokens" ADD CONSTRAINT "refresh_tokens_access_token_id_access_tokens_id_fk" FOREIGN KEY ("access_token_id") REFERENCES "public"."access_tokens"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "refresh_tokens" ADD CONSTRAINT "refresh_tokens_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "users_email_idx" ON "users" USING btree ("email");--> statement-breakpoint
CREATE INDEX "assessments_user_idx" ON "assessments" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "assessments_assessment_name_idx" ON "assessments" USING btree ("assessment_name");--> statement-breakpoint
CREATE INDEX "profiles_psychological_idx" ON "profiles" USING hnsw ("psychological_embedding" vector_cosine_ops);--> statement-breakpoint
CREATE INDEX "profiles_values_idx" ON "profiles" USING hnsw ("values_embedding" vector_cosine_ops);--> statement-breakpoint
CREATE INDEX "profiles_interests_idx" ON "profiles" USING hnsw ("interests_embedding" vector_cosine_ops);--> statement-breakpoint
CREATE INDEX "profiles_behavioral_idx" ON "profiles" USING hnsw ("behavioral_embedding" vector_cosine_ops);--> statement-breakpoint
CREATE INDEX "profiles_user_idx" ON "profiles" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "spaces_client_id_idx" ON "spaces" USING btree ("client_id");--> statement-breakpoint
CREATE INDEX "spaces_slug_idx" ON "spaces" USING btree ("slug");--> statement-breakpoint
CREATE INDEX "members_space_idx" ON "members" USING btree ("space_id");--> statement-breakpoint
CREATE INDEX "members_user_idx" ON "members" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "posts_space_idx" ON "posts" USING btree ("space_id");--> statement-breakpoint
CREATE INDEX "posts_author_idx" ON "posts" USING btree ("author_id");--> statement-breakpoint
CREATE INDEX "posts_created_at_idx" ON "posts" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "authorization_codes_code_idx" ON "authorization_codes" USING btree ("code");--> statement-breakpoint
CREATE INDEX "authorization_codes_client_idx" ON "authorization_codes" USING btree ("client_id");--> statement-breakpoint
CREATE INDEX "authorization_codes_user_idx" ON "authorization_codes" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "access_tokens_hash_idx" ON "access_tokens" USING btree ("token_hash");--> statement-breakpoint
CREATE INDEX "access_tokens_jti_idx" ON "access_tokens" USING btree ("jti");--> statement-breakpoint
CREATE INDEX "access_tokens_client_idx" ON "access_tokens" USING btree ("client_id");--> statement-breakpoint
CREATE INDEX "access_tokens_user_idx" ON "access_tokens" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "refresh_tokens_hash_idx" ON "refresh_tokens" USING btree ("token_hash");--> statement-breakpoint
CREATE INDEX "refresh_tokens_jti_idx" ON "refresh_tokens" USING btree ("jti");--> statement-breakpoint
CREATE INDEX "refresh_tokens_client_idx" ON "refresh_tokens" USING btree ("client_id");--> statement-breakpoint
CREATE INDEX "refresh_tokens_user_idx" ON "refresh_tokens" USING btree ("user_id");