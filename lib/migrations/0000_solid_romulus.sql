CREATE TYPE "public"."activity_level" AS ENUM('sedentary', 'light', 'moderate', 'active', 'very_active');--> statement-breakpoint
CREATE TYPE "public"."drinking" AS ENUM('never', 'sometimes', 'regularly');--> statement-breakpoint
CREATE TYPE "public"."education_level" AS ENUM('middle_school', 'high_school', 'bachelor', 'master', 'phd', 'vocational', 'other');--> statement-breakpoint
CREATE TYPE "public"."ethnicity" AS ENUM('white_caucasian', 'hispanic_latino', 'black_african', 'east_asian', 'south_asian', 'middle_eastern', 'pacific_islander', 'indigenous', 'mixed', 'other');--> statement-breakpoint
CREATE TYPE "public"."gender" AS ENUM('man', 'woman', 'non_binary');--> statement-breakpoint
CREATE TYPE "public"."has_children" AS ENUM('no', 'yes');--> statement-breakpoint
CREATE TYPE "public"."relationship_intent" AS ENUM('serious_relationship', 'casual_dating', 'friendship', 'chat');--> statement-breakpoint
CREATE TYPE "public"."relationship_style" AS ENUM('monogamous', 'ethical_non_monogamous', 'open', 'other');--> statement-breakpoint
CREATE TYPE "public"."religion" AS ENUM('none', 'christian', 'muslim', 'jewish', 'buddhist', 'hindu', 'spiritual', 'other');--> statement-breakpoint
CREATE TYPE "public"."sexual_orientation" AS ENUM('straight', 'gay', 'lesbian', 'bisexual', 'pansexual', 'asexual', 'queer', 'other');--> statement-breakpoint
CREATE TYPE "public"."smoking" AS ENUM('never', 'sometimes', 'regularly');--> statement-breakpoint
CREATE TYPE "public"."wants_children" AS ENUM('yes', 'no', 'open');--> statement-breakpoint
CREATE TYPE "public"."tier_interval" AS ENUM('month', 'year', 'one_time');--> statement-breakpoint
CREATE TYPE "public"."member_role" AS ENUM('owner', 'admin', 'member');--> statement-breakpoint
CREATE TYPE "public"."member_status" AS ENUM('pending', 'waiting_payment', 'active', 'suspended');--> statement-breakpoint
CREATE TYPE "public"."attendee_status" AS ENUM('going', 'interested', 'attended');--> statement-breakpoint
CREATE TYPE "public"."profile_item_type" AS ENUM('photo', 'prompt');--> statement-breakpoint
CREATE TABLE "users" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" text DEFAULT '' NOT NULL,
	"username" text,
	"email" text NOT NULL,
	"birthdate" date,
	"gender" "gender",
	"sexual_orientation" text[] DEFAULT '{}'::text[] NOT NULL,
	"height_cm" integer,
	"relationship_intent" text[] DEFAULT '{}'::text[] NOT NULL,
	"relationship_style" "relationship_style",
	"has_children" "has_children",
	"wants_children" "wants_children",
	"religion" "religion",
	"smoking" "smoking",
	"drinking" "drinking",
	"activity_level" "activity_level",
	"tags" text[] DEFAULT '{}'::text[] NOT NULL,
	"job_title" text,
	"education_level" "education_level",
	"school_name" text,
	"languages" text[] DEFAULT '{}'::text[] NOT NULL,
	"ethnicity" "ethnicity",
	"email_verified" boolean DEFAULT false NOT NULL,
	"image" text,
	"location" text,
	"coordinates" geometry(point),
	"location_updated_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "users_username_unique" UNIQUE("username"),
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE "account" (
	"id" text PRIMARY KEY NOT NULL,
	"account_id" text NOT NULL,
	"provider_id" text NOT NULL,
	"user_id" uuid NOT NULL,
	"access_token" text,
	"refresh_token" text,
	"id_token" text,
	"access_token_expires_at" timestamp,
	"refresh_token_expires_at" timestamp,
	"scope" text,
	"password" text,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "session" (
	"id" text PRIMARY KEY NOT NULL,
	"expires_at" timestamp NOT NULL,
	"token" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	"ip_address" text,
	"user_agent" text,
	"user_id" uuid NOT NULL,
	CONSTRAINT "session_token_unique" UNIQUE("token")
);
--> statement-breakpoint
CREATE TABLE "verification" (
	"id" text PRIMARY KEY NOT NULL,
	"identifier" text NOT NULL,
	"value" text NOT NULL,
	"expires_at" timestamp NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "spaces" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" text NOT NULL,
	"slug" text NOT NULL,
	"description" text,
	"image" text,
	"visibility" text DEFAULT 'public' NOT NULL,
	"type" text DEFAULT 'free' NOT NULL,
	"join_policy" text DEFAULT 'open' NOT NULL,
	"is_active" boolean DEFAULT true NOT NULL,
	"stripe_account_id" text,
	"stripe_account_enabled" boolean DEFAULT false NOT NULL,
	"owner_id" uuid NOT NULL,
	"tags" text[] DEFAULT '{}',
	"embedding" vector(1536),
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "spaces_slug_unique" UNIQUE("slug")
);
--> statement-breakpoint
CREATE TABLE "membership_tiers" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"space_id" uuid NOT NULL,
	"name" text NOT NULL,
	"description" text,
	"price" integer DEFAULT 0 NOT NULL,
	"currency" text DEFAULT 'EUR' NOT NULL,
	"interval" "tier_interval" DEFAULT 'month' NOT NULL,
	"is_active" boolean DEFAULT true NOT NULL
);
--> statement-breakpoint
CREATE TABLE "members" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"space_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"tier_id" uuid,
	"role" "member_role" DEFAULT 'member' NOT NULL,
	"status" "member_status" DEFAULT 'active' NOT NULL,
	"subscription_id" text,
	"current_period_end" timestamp,
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
CREATE TABLE "event_attendees" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"event_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"status" "attendee_status" DEFAULT 'going' NOT NULL,
	"registered_at" timestamp DEFAULT now() NOT NULL,
	"attended_at" timestamp,
	"payment_status" text,
	"stripe_checkout_session_id" text
);
--> statement-breakpoint
CREATE TABLE "events" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"space_id" uuid NOT NULL,
	"title" text NOT NULL,
	"description" text,
	"location" text,
	"coordinates" geometry(point),
	"starts_at" timestamp NOT NULL,
	"ends_at" timestamp,
	"max_attendees" integer,
	"price" integer,
	"currency" text DEFAULT 'eur',
	"tags" text[] DEFAULT '{}',
	"created_by" uuid NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "connections" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"initiator_id" uuid NOT NULL,
	"recipient_id" uuid NOT NULL,
	"target_user_item_id" uuid NOT NULL,
	"initial_message" text,
	"status" text DEFAULT 'pending' NOT NULL,
	"last_message_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "messages" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"connection_id" uuid NOT NULL,
	"sender_id" uuid NOT NULL,
	"content" text NOT NULL,
	"read_at" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "impressions" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"item_id" uuid NOT NULL,
	"item_type" text NOT NULL,
	"action" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "embeddings" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"entity_id" text NOT NULL,
	"entity_type" text NOT NULL,
	"embedding" vector(256) NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "user_items" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"type" "profile_item_type" NOT NULL,
	"prompt_key" text,
	"content" text NOT NULL,
	"display_order" integer NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "user_items_user_order_unique" UNIQUE("user_id","display_order")
);
--> statement-breakpoint
CREATE TABLE "notifications" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"type" text DEFAULT 'generic' NOT NULL,
	"text" text NOT NULL,
	"image" text,
	"href" text,
	"read" boolean DEFAULT false NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "account" ADD CONSTRAINT "account_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "session" ADD CONSTRAINT "session_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "spaces" ADD CONSTRAINT "spaces_owner_id_users_id_fk" FOREIGN KEY ("owner_id") REFERENCES "public"."users"("id") ON DELETE restrict ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "membership_tiers" ADD CONSTRAINT "membership_tiers_space_id_spaces_id_fk" FOREIGN KEY ("space_id") REFERENCES "public"."spaces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "members" ADD CONSTRAINT "members_space_id_spaces_id_fk" FOREIGN KEY ("space_id") REFERENCES "public"."spaces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "members" ADD CONSTRAINT "members_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "members" ADD CONSTRAINT "members_tier_id_membership_tiers_id_fk" FOREIGN KEY ("tier_id") REFERENCES "public"."membership_tiers"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "posts" ADD CONSTRAINT "posts_space_id_spaces_id_fk" FOREIGN KEY ("space_id") REFERENCES "public"."spaces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "posts" ADD CONSTRAINT "posts_author_id_users_id_fk" FOREIGN KEY ("author_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "event_attendees" ADD CONSTRAINT "event_attendees_event_id_events_id_fk" FOREIGN KEY ("event_id") REFERENCES "public"."events"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "event_attendees" ADD CONSTRAINT "event_attendees_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "events" ADD CONSTRAINT "events_space_id_spaces_id_fk" FOREIGN KEY ("space_id") REFERENCES "public"."spaces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "events" ADD CONSTRAINT "events_created_by_users_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "connections" ADD CONSTRAINT "connections_initiator_id_users_id_fk" FOREIGN KEY ("initiator_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "connections" ADD CONSTRAINT "connections_recipient_id_users_id_fk" FOREIGN KEY ("recipient_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "connections" ADD CONSTRAINT "connections_target_user_item_id_user_items_id_fk" FOREIGN KEY ("target_user_item_id") REFERENCES "public"."user_items"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "messages" ADD CONSTRAINT "messages_connection_id_connections_id_fk" FOREIGN KEY ("connection_id") REFERENCES "public"."connections"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "messages" ADD CONSTRAINT "messages_sender_id_users_id_fk" FOREIGN KEY ("sender_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "impressions" ADD CONSTRAINT "impressions_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "user_items" ADD CONSTRAINT "user_items_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "notifications" ADD CONSTRAINT "notifications_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "users_email_idx" ON "users" USING btree ("email");--> statement-breakpoint
CREATE INDEX "users_username_idx" ON "users" USING btree ("username");--> statement-breakpoint
CREATE INDEX "users_coordinates_gist_idx" ON "users" USING gist ("coordinates");--> statement-breakpoint
CREATE INDEX "account_userId_idx" ON "account" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "session_userId_idx" ON "session" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "verification_identifier_idx" ON "verification" USING btree ("identifier");--> statement-breakpoint
CREATE INDEX "spaces_slug_idx" ON "spaces" USING btree ("slug");--> statement-breakpoint
CREATE INDEX "spaces_tags_idx" ON "spaces" USING gin ("tags");--> statement-breakpoint
CREATE INDEX "members_space_idx" ON "members" USING btree ("space_id");--> statement-breakpoint
CREATE INDEX "members_user_idx" ON "members" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "members_tier_idx" ON "members" USING btree ("tier_id");--> statement-breakpoint
CREATE INDEX "posts_space_idx" ON "posts" USING btree ("space_id");--> statement-breakpoint
CREATE INDEX "posts_author_idx" ON "posts" USING btree ("author_id");--> statement-breakpoint
CREATE INDEX "posts_created_at_idx" ON "posts" USING btree ("created_at");--> statement-breakpoint
CREATE UNIQUE INDEX "event_attendees_event_user_idx" ON "event_attendees" USING btree ("event_id","user_id");--> statement-breakpoint
CREATE INDEX "event_attendees_user_idx" ON "event_attendees" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "events_space_idx" ON "events" USING btree ("space_id");--> statement-breakpoint
CREATE INDEX "events_starts_at_idx" ON "events" USING btree ("starts_at");--> statement-breakpoint
CREATE INDEX "events_created_by_idx" ON "events" USING btree ("created_by");--> statement-breakpoint
CREATE INDEX "events_coordinates_gist_idx" ON "events" USING gist ("coordinates");--> statement-breakpoint
CREATE INDEX "events_tags_idx" ON "events" USING gin ("tags");--> statement-breakpoint
CREATE INDEX "connections_initiator_idx" ON "connections" USING btree ("initiator_id");--> statement-breakpoint
CREATE INDEX "connections_recipient_idx" ON "connections" USING btree ("recipient_id");--> statement-breakpoint
CREATE INDEX "connections_status_idx" ON "connections" USING btree ("status");--> statement-breakpoint
CREATE UNIQUE INDEX "connections_participants_uidx" ON "connections" USING btree ("initiator_id","recipient_id");--> statement-breakpoint
CREATE INDEX "messages_connection_idx" ON "messages" USING btree ("connection_id");--> statement-breakpoint
CREATE INDEX "messages_created_at_idx" ON "messages" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "impressions_user_idx" ON "impressions" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "impressions_item_idx" ON "impressions" USING btree ("item_id");--> statement-breakpoint
CREATE INDEX "impressions_user_item_idx" ON "impressions" USING btree ("user_id","item_id");--> statement-breakpoint
CREATE INDEX "impressions_created_at_idx" ON "impressions" USING btree ("created_at");--> statement-breakpoint
CREATE UNIQUE INDEX "embeddings_entity_idx" ON "embeddings" USING btree ("entity_id","entity_type");--> statement-breakpoint
CREATE INDEX "embeddings_type_idx" ON "embeddings" USING btree ("entity_type");--> statement-breakpoint
CREATE INDEX "embeddings_hnsw_idx" ON "embeddings" USING hnsw ("embedding" vector_cosine_ops);--> statement-breakpoint
CREATE INDEX "user_items_user_idx" ON "user_items" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "user_items_order_idx" ON "user_items" USING btree ("user_id","display_order");--> statement-breakpoint
CREATE INDEX "notifications_user_idx" ON "notifications" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "notifications_user_read_idx" ON "notifications" USING btree ("user_id","read");--> statement-breakpoint
CREATE INDEX "notifications_created_at_idx" ON "notifications" USING btree ("created_at");