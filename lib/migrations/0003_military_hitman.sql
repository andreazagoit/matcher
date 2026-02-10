ALTER TABLE "daily_matches" DROP CONSTRAINT "daily_matches_user_id_match_id_date_pk";--> statement-breakpoint
ALTER TABLE "daily_matches" ADD CONSTRAINT "daily_matches_user_id_match_id_pk" PRIMARY KEY("user_id","match_id");--> statement-breakpoint
ALTER TABLE "daily_matches" DROP COLUMN "date";