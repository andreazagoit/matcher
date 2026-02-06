CREATE TYPE "public"."tier_interval" AS ENUM('month', 'year', 'one_time');--> statement-breakpoint
ALTER TYPE "public"."member_status" ADD VALUE 'waiting_payment' BEFORE 'active';--> statement-breakpoint
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
ALTER TABLE "spaces" ADD COLUMN "type" text DEFAULT 'free' NOT NULL;--> statement-breakpoint
ALTER TABLE "members" ADD COLUMN "tier_id" uuid;--> statement-breakpoint
ALTER TABLE "members" ADD COLUMN "subscription_id" text;--> statement-breakpoint
ALTER TABLE "members" ADD COLUMN "current_period_end" timestamp;--> statement-breakpoint
ALTER TABLE "membership_tiers" ADD CONSTRAINT "membership_tiers_space_id_spaces_id_fk" FOREIGN KEY ("space_id") REFERENCES "public"."spaces"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "members" ADD CONSTRAINT "members_tier_id_membership_tiers_id_fk" FOREIGN KEY ("tier_id") REFERENCES "public"."membership_tiers"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "members_tier_idx" ON "members" USING btree ("tier_id");