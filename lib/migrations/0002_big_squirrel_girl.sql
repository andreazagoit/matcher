CREATE TABLE "connections" (
	"requester_id" uuid NOT NULL,
	"target_id" uuid NOT NULL,
	"status" text DEFAULT 'interested' NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "connections_requester_id_target_id_pk" PRIMARY KEY("requester_id","target_id")
);
--> statement-breakpoint
CREATE TABLE "daily_matches" (
	"user_id" uuid NOT NULL,
	"match_id" uuid NOT NULL,
	"date" date NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "daily_matches_user_id_match_id_date_pk" PRIMARY KEY("user_id","match_id","date")
);
--> statement-breakpoint
ALTER TABLE "connections" ADD CONSTRAINT "connections_requester_id_users_id_fk" FOREIGN KEY ("requester_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "connections" ADD CONSTRAINT "connections_target_id_users_id_fk" FOREIGN KEY ("target_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_matches" ADD CONSTRAINT "daily_matches_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_matches" ADD CONSTRAINT "daily_matches_match_id_users_id_fk" FOREIGN KEY ("match_id") REFERENCES "public"."users"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "connections_requester_idx" ON "connections" USING btree ("requester_id");--> statement-breakpoint
CREATE INDEX "connections_target_idx" ON "connections" USING btree ("target_id");--> statement-breakpoint
CREATE INDEX "connections_status_idx" ON "connections" USING btree ("status");